import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=[skip.size(2), skip.size(3)], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
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
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x   # x2

