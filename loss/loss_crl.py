from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


class CRLLoss(nn.Module):
    def __init__(self, 
                ss_epoch, 
                ratio=0.5,
                temperature=0.5,
                w=0.5
                ):
        super(CRLLoss, self).__init__()
        self.ss_epoch = ss_epoch
        self.ratio = ratio
        self.T = temperature
        self.w = w

        self.loss_1 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.loss_2 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.loss_3 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.gt_ce = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.ps_ce = SoftCrossEntropyLoss(reduction='mean')
    
    def forward(self, preds1, preds2, preds3, target, epoch):
        if epoch < self.ss_epoch:
            n, h, w = target.size()
            mask_bg = (target==0) + (target==1)
            mask_fg = (target==2) + (target==3)
            loss_1 = self.loss_1(preds1, target)
            loss_2 = self.loss_2(preds2, target)
            loss_3 = self.loss_3(preds3, target)

            loss_fg_1 = loss_1[mask_fg]
            loss_fg_2 = loss_2[mask_fg]
            loss_fg_3 = loss_3[mask_fg]
            mu_1 = loss_fg_1.data + torch.abs(loss_fg_2.data - loss_fg_3.data)
            mu_2 = loss_fg_2.data + torch.abs(loss_fg_3.data - loss_fg_1.data)
            mu_3 = loss_fg_3.data + torch.abs(loss_fg_1.data - loss_fg_2.data)
            ind_1_sorted = torch.argsort(mu_1)
            ind_2_sorted = torch.argsort(mu_2)
            ind_3_sorted = torch.argsort(mu_3)
            
            num_remember = int(len(ind_1_sorted) * self.ratio)
            num = n*h*w - len(ind_1_sorted) + num_remember
            loss_1_update = (torch.sum(loss_fg_1[ind_1_sorted[:num_remember]]) + torch.sum(loss_1[mask_bg])) / num
            loss_2_update = (torch.sum(loss_fg_2[ind_2_sorted[:num_remember]]) + torch.sum(loss_2[mask_bg])) / num
            loss_3_update = (torch.sum(loss_fg_3[ind_3_sorted[:num_remember]]) + torch.sum(loss_3[mask_bg])) / num

            return loss_1_update, loss_2_update, loss_3_update
        else:
            weight = epoch/120*(1-self.w) + self.w

            preds = (preds1.clone().detach() + preds2.clone().detach() + preds3.clone().detach()) / 3
            preds = torch.pow(preds, 1/self.T)
            pseudo = preds / torch.sum(preds, dim=1, keepdim=True)
            
            loss_1 = weight * self.ps_ce(preds1, pseudo) +  (1-weight) * self.gt_ce(preds1, target) 
            loss_2 = weight * self.ps_ce(preds2, pseudo) +  (1-weight) * self.gt_ce(preds2, target) 
            loss_3 = weight * self.ps_ce(preds3, pseudo) +  (1-weight) * self.gt_ce(preds3, target) 

            return loss_1, loss_2, loss_3


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SoftCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = -F.log_softmax(inputs, dim=1)
        loss = torch.sum(inputs * targets, dim=1)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)

        return loss
