from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


class BapMTLoss(nn.Module):
    """
    Hybrid Loss for BapnetMT: 
        Segmentation Loss: seg loss + seg pseudo seg loss + sim seg pseudo loss
        Classification Loss: cls loss
        Consistency Loss: seg consistency loss + sim consisitency loss + seg-sim consistency
    """
    def __init__(self,
                alpha = 1.0,  # cls loss weight
                beta = 1,  # cons loss weight
                w = 0.5,
                use_curriculum = True,
                use_seg_sim_cons = True,
                aux_params: Optional[dict] = None,
                ):
        super(BapMTLoss, self).__init__()
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.cons_seg_loss = softmax_kl_loss
        self.cons_loss = nn.MSELoss(reduction='mean')
    
        self.alpha = alpha
        self.beta = beta

        self.use_curriculum = use_curriculum
        self.use_seg_sim_cons = use_seg_sim_cons
        self.T = 120 
        self.w = w 

    def forward(self, gt_label, epoch, **outputs):
        seg_feat_q = outputs["seg_feat_q"]

        w = epoch/self.T*(1-self.w) + self.w
        seg_gt_term = self.seg_loss(seg_feat_q, gt_label)
        seg_sim_term = self.seg_loss(seg_feat_q, outputs["sim_pseudo_label"])
        seg_seg_term = self.seg_loss(seg_feat_q, outputs["seg_pseudo_label"])
        seg_term = w*(seg_sim_term+seg_seg_term)/2 + (1-w)*seg_gt_term 

        cls_term = self.cls_loss(outputs["cls_feat"], outputs["cls_label"])

        cons_seg_term = self.cons_seg_loss(seg_feat_q, outputs["seg_feat_k"])
        cons_sim_term = self.cons_loss(outputs["sim_q"], outputs["sim_k"])
        if self.use_seg_sim_cons:
            seg_sim_feat = F.softmax(seg_feat_q.clone().detach(), dim=1)[:, 1, ...]
            seg_sim_feat /= seg_sim_feat.max()
            cons_seg_sim_term = self.cons_loss(seg_sim_feat, outputs["sim_q"])
            cons_term = (cons_seg_term+cons_sim_term+cons_seg_sim_term) / 3
        else:
            cons_term = (cons_seg_term+cons_sim_term) / 2

        loss = seg_term + self.alpha * cls_term + self.beta * cons_term

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






