from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


class UnetWithCLSLoss(nn.Module):
    """
    Hybrid Loss for BapnetMT: 
        Segmentation Loss: seg loss + seg pseudo seg loss + sim seg pseudo loss
        Classification Loss: cls loss
        Consistency Loss: seg consistency loss + sim consisitency loss + seg-sim consistency
    """
    def __init__(self,
                alpha = 1.0,  # cls loss weight
                aux_params: Optional[dict] = None,
                ):
        super(UnetWithCLSLoss, self).__init__()
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    
        self.alpha = alpha

    def forward(self, seg_feat, seg_label, cls_feat, cls_label):
        seg_term = self.seg_loss(seg_feat, seg_label)
        cls_term = self.cls_loss(cls_feat, cls_label)

        loss = seg_term + self.alpha * cls_term 
        loss_term = {"seg_term":seg_term.item(), "cls_term":cls_term.item()}

        return loss, loss_term






