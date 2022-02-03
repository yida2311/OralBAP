from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


class SegMTLoss(nn.Module):
    """
    Hybrid loss for UnetMT
        Segmentation Loss: gt seg loss + pseudo seg loss
        Consistency Loss: seg cons loss
    """
    def __init__(self,
                alpha = 1.0,
                w = 0.5,
                use_curriculum = False,
                use_kl = True,
                aux_params: Optional[dict] = None,
                ):
        super(SegMTLoss, self).__init__()
        self.gt_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if use_kl:
            self.cons_loss = softmax_kl_loss
        else:
            self.cons_loss = softmax_mse_loss
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.alpha = alpha
        self.use_curriculum = use_curriculum
        self.T = 120 
        self.w = w
    
    def forward(self, feat, mask, teacher_feat, pseudo, epoch):
        loss = self.seg_loss(feat, pseudo)
        if self.use_curriculum:
            gt_term = self.gt_loss(feat, mask)
            w = epoch /self.T * (1-self.w) + self.w
            loss =  w* loss + (1-w) * gt_term

        loss += self.alpha * self.cons_loss(feat, teacher_feat)

        return loss


class SegTALoss(nn.Module):
    """
    loss for UnetTA
        Segmentation Loss: gt seg loss + pseudo seg loss
    """
    def __init__(self,
                alpha = 1.0,
                w = 0.5,
                use_curriculum = False,
                aux_params: Optional[dict] = None,
                ):
        super(SegTALoss, self).__init__()
        self.gt_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.alpha = alpha
        self.use_curriculum = use_curriculum
        self.T = 120 
        self.w = w
    
    def forward(self, feat, mask, teacher_feat, pseudo, epoch):
        loss = self.seg_loss(feat, pseudo)
        if self.use_curriculum:
            gt_term = self.alpha * self.gt_loss(feat, mask)
            w = epoch/self.T*(1-self.w) + self.w
            loss =  w* loss + (1-w) * gt_term

        return loss


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean. 
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=True) / num_classes
    
def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    return F.kl_div(input_log_softmax, target_softmax, size_average=True)