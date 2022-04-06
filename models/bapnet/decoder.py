from torch import nn

from ..unet.decoder import CenterBlock, DecoderBlock

class BAPnetDecoder(nn.Module):
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


