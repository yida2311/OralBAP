import random
from typing import Optional, Union, List
import torch
from torch import nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead, SegmentationModel, ClassificationHead
import segmentation_models_pytorch.base.initialization as init

from .encoder import BAPnetDecoder, BAPnetEncoder

class BAPnetTA(SegmentationModel):
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
        self.encoder_q = BAPnetEncoder(
            encoder_name,
            in_channels=in_channels, # 3
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            center=True,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
        )
        self.encoder_k = BAPnetEncoder(
            encoder_name,
            in_channels=in_channels, # 3
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            center=True,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
        )
        self.decoder = BAPnetDecoder(
            use_batchnorm=decoder_use_batchnorm,
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
        self.temperature = aux_params['temperature']
        self.weight_type = aux_params['weight_type']
        self.register_buffer("thresh", torch.zeros(1, dtype=torch.float))
        # create Background Memory Banck
        self.K = aux_params['memory_bank']['K']  # 1000
        self.T = aux_params['memory_bank']['T'] 
        self.m = aux_params['memory_bank']['m']
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.n_class = classes
        self.name = "bap-{}".format(encoder_name)
        self.initialize()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) # initialize
            param_k.requires_grad = False
    
    def initialize(self):
        init.initialize_head(self.segmentation_head)
        init.initialize_head(self.classification_head)
        init.initialize_decoder(self.encoder_q.center)
        init.initialize_decoder(self.encoder_q.decoder_block_1)
        init.initialize_decoder(self.encoder_q.decoder_block_2)
        init.initialize_decoder(self.encoder_q.proto_block)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

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
    
    def background_prototype_generation(self, feature, mask):
        n, c, h, w = feature.size()
        mask = F.interpolate(mask.unsqueeze(1).to(torch.float), size=(h, w), mode='nearest').squeeze(1).to(torch.long)
        bg_mask = mask == 1
        area = torch.sum(bg_mask, dim=(1,2)).float()
        digit = torch.sum(feature*bg_mask.unsqueeze(1), dim=(2,3)) / (area.unsqueeze(1)+1) # N x 256
        ratio = area / (h*w)
        proto_state = ratio>self.min_ratio
        proto = F.normalize(digit[proto_state], dim=1)

        return proto, proto_state

    def prototype_generation(self, feature, sim, mask):
        """
            feature: N x 256 x h x w
            sim: N x H x W
            mask: N x H x W
            min_area: constant

            return:
                digit: 4N x 256
                label: 4N
                proto: M x 256
        """
        ## mask operation
        n, c, h, w = feature.size()
        mask = F.interpolate(mask.unsqueeze(1).to(torch.float), size=(h, w), mode='nearest').squeeze(1).to(torch.long)
        # sim rectify
        sim[mask==1] = 1
        sim[mask!=1] = 1 - sim[mask!=1]
        # one hot
        mask = one_hot(mask, self.n_class) # N x 4 x h x w
        ## reweighting
        weighted_feat = feature * (sim.unsqueeze(1))
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
        # proto = F.normalize(digit[:,1,:][label[:,1]==1], dim=1)
        digit = digit.view(-1, c)
        label = label.view(-1)

        return digit, label
    
    def prototype_selection(self):
        inds = list(range(self.K))
        random.shuffle(inds)
        inds = torch.tensor(inds[:self.T], dtype=torch.long)
        proto = self.queue[:, inds]

        return proto  # [256 x 100]
    
    def similarity_weight(self, proto, proto_k, proto_k_state):
        """
         proto: (256 x (M+K)
         proto_k: M x 256
         proto_k_state: N

         return:
            weight: N x (M+K)
        """
        _, k = proto.size()
        m, _ = proto_k.size()
        n = proto_k_state.size(0)
        weight = torch.ones((n, k), dtype=torch.float).cuda()
        weight[proto_k_state] = torch.mm(proto_k, proto)
        if self.weight_type == 'softmax':
            weight = F.softmax(weight/self.temperature, dim=1)
        elif self.weight_type == 'weighted':
            weight = weight / torch.sum(weight, dim=1).unsqueeze(1)
        elif self.weight_type == 'mean':
            weight = torch.ones((n, k), dtype=torch.float).cuda() / k

        return weight

    def similarity_calculation(self, feature, proto, weight):
        """
            feature: N x 256 x h x w
            proto: 256 x k
            weight: N x k
            return:
                sim: N x h x w
        """
        n, c, h, w = feature.size()
        k = proto.size(1)
        # normalize
        feature = F.normalize(feature, dim=1)
        # similarity matrix
        feature = feature.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (Nhw) x 256
        sim = F.relu(torch.mm(feature, proto))  # (Nhw) x k
        sim = sim.view(n, h, w, k).permute(0, 3, 1, 2).contiguous() * weight.unsqueeze(2).unsqueeze(3) # N x k x h x w
        sim = torch.sum(sim, dim=1)

        return sim.detach()
    
    def multi_similarity_calculation(self, feature, proto):
        n, c, h, w = feature.size()
        k = proto.size(1)
        # normalize
        feature = F.normalize(feature, dim=1)
        # similarity matrix
        feature = feature.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (Nhw) x 256
        sim = F.relu(torch.mm(feature, proto))  # (Nhw) x k
        sim = sim.view(n, h, w, k).permute(0, 3, 1, 2).contiguous()

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
        bg_sim = sim * bg_mask

        threshs = torch.sum(bg_sim, dim=(1,2)) / (torch.sum(bg_mask, dim=(1,2))+1e-1)  # N
        thresh = threshs.mean()
        self.thresh = (1-self.momentum) * thresh + self.momentum * self.thresh
        ratio = torch.true_divide(torch.sum(bg_mask, dim=(1,2)), (H*W))
        threshs[ratio<self.min_ratio] = self.thresh

        noise_mask = (sim > threshs.unsqueeze(1).unsqueeze(2)) & fg_mask
        pseudo[noise_mask] = 1  # modify to normal

        return pseudo
        
    def forward(self, img, mask):
        _, H, W = mask.size()
        # query forward
        encoder_feats, feat_q = self.encoder_q(img) # [x8,x4,x2] , x8[256] 
        seg_feat = self.decoder(*encoder_feats) # x2[64]
        # key forward
        with torch.no_grad():
            self._momentum_update_key_encoder()
            feat_k = self.encoder_k.get_proto(img) # x8[256]
            proto_k, proto_k_state = self.background_prototype_generation(feat_k, mask) # M x 256, N 
        # # proto selection for similarity calculation
        proto = self.prototype_selection() #  100 x 256
        # bg proto construction
        # proto = torch.cat([proto_k.T, self.queue], dim=1) # 256 x (M+K) 
        # similarity map weight
        sim_weight = self.similarity_weight(proto, proto_k, proto_k_state)
        # similarity calculation
        sim = self.similarity_calculation(feat_q, proto, sim_weight) # n x h x w
        # pseudo mask generation for segmentation
        seg_label = self.pseudo_mask_generation(sim, mask)
        # classification digit & label generation
        digit, cls_label = self.prototype_generation(feat_q, sim, mask) # nx4x256, nx4
        # sgementation head
        seg_feat = self.segmentation_head(seg_feat)
        # linear classifier
        cls_feat = self.classification_head(digit)

        sim = F.interpolate(sim.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)
        self._dequeue_and_enqueue(proto_k)

        return seg_feat, seg_label, cls_feat, cls_label, sim

    def inference(self, img):
        encoder_feats, _ = self.encoder_q(img) # [x8,x4,x2] , x8[256] 
        seg_feat = self.decoder(*encoder_feats) # x2[64]
        seg_feat = self.segmentation_head(seg_feat)
        
        return seg_feat
    
    def get_similarity_map(self, img, mask):
        _, H, W = mask.size()
        # query forward
        _, feat_q = self.encoder_q(img) # [x8,x4,x2] , x8[256] 
        # key forward
        with torch.no_grad():
            feat_k = self.encoder_k.get_proto(img) # x8[256]
            proto_k, proto_k_state = self.background_prototype_generation(feat_k, mask) # M x 256, N 
        # bg proto construction
        proto = self.prototype_selection() # 100 x 256
        # similarity map weight
        sim_weight = self.similarity_weight(proto, proto_k, proto_k_state)
        # similarity calculation
        sim = self.similarity_calculation(feat_q, proto, sim_weight) # n x h x w
        # pseudo mask generation
        pseudo = self.pseudo_mask_generation(sim, mask)
        sim = F.interpolate(sim.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)

        return sim, pseudo
    
    def get_multi_similarity_map(self, img, mask):
        _, H, W = mask.size()
        # query forward
        _, feat_q = self.encoder_q(img) # [x8,x4,x2] , x8[256] 
        # key forward
        with torch.no_grad():
            feat_k = self.encoder_k.get_proto(img) # x8[256]
            proto_k, proto_k_state = self.background_prototype_generation(feat_k, mask) # M x 256, N 
        # bg proto construction
        proto = self.prototype_selection() # 100 x 256
        # similarity map weight
        sim_weight = self.similarity_weight(proto, proto_k, proto_k_state)
        # print(sim_weight)
        # similarity calculation
        sim = self.similarity_calculation(feat_q, proto, sim_weight) # n x h x w
        sim = F.interpolate(sim.unsqueeze(1), size=(H,W), mode='bilinear').squeeze()
        # similarity calculation
        sims = self.multi_similarity_calculation(feat_q, proto) # n x k x h x w
        sims = F.interpolate(sims, size=(H,W), mode='bilinear')

        return sims, sim
        

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









