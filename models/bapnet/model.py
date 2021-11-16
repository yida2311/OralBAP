import random
from typing import Optional, Union, List
import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead, SegmentationModel, ClassificationHead

from .decoder import BAPnetDecoder

class BAPnet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 64),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 4,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels, # 3
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = BAPnetDecoder(
            encoder_channels=self.encoder.out_channels,  # [3,64,64,128,256,512]
            decoder_channels=decoder_channels, # [256, 128, 64, 64]
            n_blocks=encoder_depth-1, # 4
            use_batchnorm=decoder_use_batchnorm,
            center=True, # attention\conv\Identity
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1], # 64
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        self.dim = 256
        self.classification_head = nn.Linear(decoder_channels[0], classes, bias=False)
        
        self.min_ratio = aux_params['min_ratio']
        self.momentum = aux_params['momentum']
        self.register_buffer("thresh", torch.zeros(1, dtype=torch.float))
        # create Background Memory Banck
        self.K = aux_params['memory_bank']['K']  # 1000
        self.T = aux_params['memory_bank']['T']  # 100
        self.item = 1  # [normal, mucosa, tumor]
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.n_class = classes
        self.name = "bap-{}".format(encoder_name)
        self.initialize()
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        " keys: batch_size x dim "
        keys = F.normalize(keys, dim=1)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            self.queue[:, ptr:] = keys[:(self.K-ptr), :].T
            self.queue[:, :(ptr+batch_size-self.K)] = keys[(self.K-ptr):, :].T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr


    def prototype_generation(self, feature, sim, mask):
        """
            feature: N x 256 x h x w
            sim: N x H x W
            mask: N x H x W
            min_area: constant

            return:
                digit: 4N x 256
                label: 4N
                proto: [M1 x 256, M2 x 256, M3 x 256]
        """
        ## mask operation
        n, c, h, w = feature.size()
        mask = F.interpolate(mask.unsqueeze(1).to(torch.float), size=(h, w), mode='nearest').squeeze(1).to(torch.long)
        mask = one_hot(mask, self.n_class) # N x 4 x h x w
        ## reweighting
        weighted_feat = feature * (1-sim.unsqueeze(1))
        digit = weighted_feat.unsqueeze(1) * mask.unsqueeze(2)  # N x 4 x 256 x h x w
        # GAP
        weight = torch.sum(mask*sim.unsqueeze(1), dim=(2,3))
        digit = torch.sum(digit, dim=(3,4)) / (weight.unsqueeze(2) + 1) # N x 4 x 256
        # filter out specific class label, if area < min_area
        label = torch.tensor(range(self.n_class), dtype=torch.long)
        label = label.repeat(n, 1).cuda()  # N x 4
        area = torch.sum(mask, dim=(2,3)) # N x 4
        ratio = area / (h*w)
        label[ratio<self.min_ratio] = -1
        proto = F.normalize(digit[:,1,:][label[:,1]==1], dim=1)
        digit = digit.view(-1, c)
        label = label.view(-1)

        return digit, label, proto
    
    def prototype_selection(self):
        inds = list(range(self.K))
        random.shuffle(inds)
        inds = torch.tensor(inds[:self.T], dtype=torch.long)
        proto = self.queue[:, inds]

        return proto  # [256 x 100]

    def similarity_calculation(self, feature, proto):
        """
            feature: N x 256 x h x w
            proto: 256 x k
            mask: N x H x W
            return:
                sim: N x 3 x h x w
        """
        n, c, h, w = feature.size()
        # normalize
        feature = F.normalize(feature, dim=1)
        # similarity matrix
        feature = feature.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (Nhw) x 256
        sim = F.relu(torch.mm(feature, proto))  # (Nhw) x k
        sim = torch.mean(sim, dim=1).view(n, h, w)

        return sim.detach()
    
    def pseudo_mask_generation(self, sim, mask):
        """ if sim > high_thresh, it means these pixels are similar to background;
            if sim < low_thresh, it means these pixels remain formal label;
            else, it means uncertain
            sim: N x h x w
            mask: N x H x W

            return:
                pseudo    
        """
        n, H, W = mask.size()
        pseudo = mask.clone().detach()
        sim = F.interpolate(sim.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)
        bg_mask = (mask == 1)
        fg_mask = (mask != 1)
        bg_sim = sim[bg_mask]
        thresh = bg_sim.sum() / bg_mask.sum()
        self.thresh = (1-self.momentum) * thresh + self.momentum * self.thresh
        noise_mask = (sim > self.thresh) & fg_mask
        pseudo[noise_mask] = 1  # modify to normal

        return pseudo
        
    def forward(self, img, mask):
        _, H, W = mask.size()
        encoder_feats = self.encoder(img) # [x1,x2,x4,x8,x16,x32]
        seg_feat, feat = self.decoder(*encoder_feats) # x2[64], x16[256] 
        # proto selection for similarity calculation
        proto = self.prototype_selection() #  100 x 256
        # similarity calculation
        sim = self.similarity_calculation(feat, proto) # n x h x w
        # pseudo mask generation
        seg_label = self.pseudo_mask_generation(sim, mask)
        # prototype generation
        digit, cls_label, new_proto = self.prototype_generation(feat, sim, mask) # nx4x256, nx4
        self._dequeue_and_enqueue(new_proto)
        # sgementation head
        seg_feat = self.segmentation_head(seg_feat)
        # linear classifier
        cls_feat = self.classification_head(digit)
        sim = F.interpolate(sim.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)
        
        return seg_feat, seg_label, cls_feat, cls_label, sim

    def inference(self, img):
        encoder_feats = self.encoder(img) # [x1,x2,x4,x8,x16,x32]
        seg_feat, _ = self.decoder(*encoder_feats) # x2[64], x16[256] 
        seg_feat = self.segmentation_head(seg_feat)
        
        return seg_feat
    
    def get_similarity_map(self, img, mask):
        _, H, W = mask.size()
        encoder_feats = self.encoder(img) # [x1,x2,x4,x8,x16,x32]
        _, feat = self.decoder(*encoder_feats) # x2[64], x16[256] 
        # proto selection for similarity calculation
        proto = self.prototype_selection() # 100 x 256
        # similarity calculation
        sim = self.similarity_calculation(feat, proto) # n x h x w
        # pseudo mask generation
        pseudo = self.pseudo_mask_generation(sim, mask)
        sim = F.interpolate(sim.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)

        return sim, pseudo
        

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def one_hot(index, classes):
    # index is not flattened (pypass ignore) ############
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]
    #####################################################
    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones) 









