from torch import nn 

from segmentation_models_pytorch.encoders import get_encoder

from ..unet.decoder import CenterBlock, DecoderBlock

class BAPnetEncoder(nn.Module):
    def __init__(self,
                encoder_name: str = "resnet34",
                in_channels: int = 3,
                encoder_depth: int = 5,
                encoder_weights: str = "imagenet",
                center = True,
                use_batchnorm=True,
                attention_type=None,
                ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels, # 3
            depth=encoder_depth,
            weights=encoder_weights,
        )
        encoder_channels = self.encoder.out_channels[::-1]  # [512, 256, 128, 64, 64, 3]
        head_channels = encoder_channels[0]

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()
        
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.decoder_block_1 = DecoderBlock(512, 256, 256, **kwargs)
        self.decoder_block_2 = DecoderBlock(256, 128, 128, **kwargs)
        self.proto_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )
        self.encoder_channels = encoder_channels[2:]
    
    def forward(self, img):
        feats = self.encoder(img)[1:][::-1]
        x = self.center(feats[0]) # 512
        x = self.decoder_block_1(x, feats[1]) # 256
        x = self.decoder_block_2(x, feats[2]) # 128
        proto = self.proto_block(x) # 256
        out_feats = [x] + feats[3:] # [128, 64, 64]

        return out_feats, proto
    
    def get_proto(self, img):
        feats = self.encoder(img)[1:][::-1]
        x = self.center(feats[0]) # 512
        x = self.decoder_block_1(x, feats[1]) # 256
        x = self.decoder_block_2(x, feats[2]) # 128
        x = self.proto_block(x) # 256
        return x


class BAPnetDecoder(nn.Module):
    def __init__(
            self,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.decoder_block_3 = DecoderBlock(128, 64, 64, **kwargs)
        self.decoder_block_4 = DecoderBlock(64, 64, 64, **kwargs)
    
    def forward(self, *features):
        x = self.decoder_block_3(features[0], features[1])
        x = self.decoder_block_4(x, features[2])
        return x