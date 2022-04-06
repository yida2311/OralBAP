from typing import Optional, Union, List
from .decoder import UnetDecoder, CenterBlock, DecoderBlock
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead, SegmentationModel, ClassificationHead

from torch import nn 
import torch
import torch.nn.functional as F

class UnetWithCLS(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

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

        self.decoder = UnetDecoderWithCLS(
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

        self.n_class = classes
        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def prototype_generation(self, feature, mask):
        """
            feature: N x 256 x h x w
            mask: N x H x W

            return:
                digit: 4N x 256
                label: 4N  
        """
        ## mask operation
        n, c, h, w = feature.size()
        mask = F.interpolate(mask.unsqueeze(1).to(torch.float), size=(h, w), mode='nearest').squeeze(1).to(torch.long)
        # one hot
        mask = one_hot(mask, self.n_class) # N x 4 x h x w
        digit = feature.unsqueeze(1) * mask.unsqueeze(2)
        weight = torch.sum(mask, dim=(2,3))
        digit = torch.sum(digit, dim=(3,4)) / (weight.unsqueeze(2)+1)
        ratio = weight.float() / (h*w)
        label = torch.tensor(range(self.n_class), dtype=torch.long)
        label = label.repeat(n, 1).cuda()  # N x 4
        label[ratio<self.min_ratio] = -1
        digit = digit.view(-1, c)
        label = label.view(-1)

        return digit, label

    
    def forward(self, img, mask):
        _, H, W = mask.size()
        encoder_feats = self.encoder(img) # [x1,x2,x4,x8,x16,x32]
        seg_feat, feat = self.decoder(*encoder_feats) # x2[64], x16[256] 
        # prototype generation
        digit, cls_label = self.prototype_generation(feat, mask) # nx4x256, nx4
        # sgementation head
        seg_feat = self.segmentation_head(seg_feat)
        seg_feat = F.interpolate(seg_feat, size=(H, W), mode='bilinear')
        # linear classifier
        cls_feat = self.classification_head(digit)

        return seg_feat, cls_feat, cls_label
    
    def inference(self, img):
        _, _, H, W = img.size()
        encoder_feats = self.encoder(img) # [x1,x2,x4,x8,x16,x32]
        seg_feat, _ = self.decoder(*encoder_feats) # x2[64], x16[256] 
        seg_feat = self.segmentation_head(seg_feat)
        seg_feat = F.interpolate(seg_feat, size=(H, W), mode='bilinear')

        return seg_feat






class UnetDecoderWithCLS(nn.Module):
    def __init__(
            self,
            encoder_channels, # for resnet34, encoder_channels=[3,64,64,128,256,512]
            decoder_channels, # (256, 128, 64, 32)
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]  # 512
        in_channels = [head_channels] + list(decoder_channels[:-1])  # [512,256,128,64]
        skip_channels = list(encoder_channels[1:]) # [256,128,64,64]
        out_channels = decoder_channels # (256, 128, 64, 64)

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        #### proto branch for classification and similarity calculation
        self.inds = 1
        self.proto_branch = nn.Sequential(
            nn.Conv2d(in_channels=out_channels[self.inds], out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )

    def extract_features(self, *features):
        head = features[0] # x32
        skips = features[1:] # [x16, x8, x4, x2]

        x = self.center(head)
        outputs = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            outputs.append(x)
        
        return outputs

    def forward(self, *features): 
        features = features[1:][::-1]# features: [c5, c4, c3, c2, c1]
        
        head = features[0] # x32
        skips = features[1:] # [x16, x8, x4, x2]
    
        x = self.center(head)
        proto = None
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            if i == self.inds:
                proto = x # x8, 128 channel
        
        proto = self.proto_branch(proto)

        return x, proto   # x2, x16


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