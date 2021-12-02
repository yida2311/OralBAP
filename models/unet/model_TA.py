import random
from typing import Optional, Union, List
import torch
from torch import nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead, SegmentationModel, ClassificationHead
import segmentation_models_pytorch.base.initialization as init

from .model import Unet

class UnetTA(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 64),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
    
        self.teacher = Unet(encoder_name=encoder_name,
                            encoder_depth=5,
                            encoder_weights=encoder_weights,
                            decoder_use_batchnorm=decoder_use_batchnorm,
                            decoder_channels=decoder_channels,
                            decoder_attention_type=decoder_attention_type,
                            in_channels=in_channels,
                            classes=classes,
                            activation=activation,
                            )
        
        self.student = Unet(encoder_name=encoder_name,
                            encoder_depth=5,
                            encoder_weights=encoder_weights,
                            decoder_use_batchnorm=decoder_use_batchnorm,
                            decoder_channels=decoder_channels,
                            decoder_attention_type=decoder_attention_type,
                            in_channels=in_channels,
                            classes=classes,
                            activation=activation,
                            )
        
        self.mt = aux_params['momentum']
        self.n_class = classes
        self.name = "unetTA-{}".format(encoder_name)

        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.copy_(param_q.data) # initialize
            param_k.requires_grad = False
        
    @torch.no_grad()
    def _momentum_update_teacher_model(self):
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data = param_k.data * self.mt + param_q.data * (1-self.mt) # initialize
    

    def pseudo_mask_generation(self, teacher_output, mask):
        n, H, W = mask.size()
        teacher_output = F.interpolate(teacher_output, size=(H, W), mode='bilinear')
        pseudo = torch.argmax(teacher_output, dim=1)
        pseudo[mask==0] = 0
        pseudo[mask==1] = 1

        return pseudo
    
    def forward(self, img, mask):
        _, H, W = mask.size()
        teacher_output = self.teacher(img)
        student_output = self.student(img)
        pseudo = self.pseudo_mask_generation(teacher_output, mask)

        self._momentum_update_teacher_model()

        return student_output, teacher_output, pseudo

    def inference(self, img):
        return self.student(img)


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


