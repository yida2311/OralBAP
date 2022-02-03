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
        # self.ps_ce = SoftCrossEntropyLoss(reduction='mean')
        self.ps_ce = nn.MSELoss(reduction='mean')
    
    def forward(self, preds1, preds2, preds3, target, epoch):
        n, h, w = target.size()
        mask_bg = (target==0) + (target==1)
        mask_fg = (target==2) + (target==3)
        loss_1 = self.loss_1(preds1, target)
        loss_2 = self.loss_2(preds2, target)
        loss_3 = self.loss_3(preds3, target)

        loss_fg_1 = loss_1[mask_fg]
        loss_fg_2 = loss_2[mask_fg]
        loss_fg_3 = loss_3[mask_fg]

        # m1 = torch.mean(loss_fg_1.data)
        # n1 = torch.mean(torch.abs(loss_fg_2.data - loss_fg_3.data))
        # m2 = torch.mean(loss_fg_2.data)
        # n2 = torch.mean(torch.abs(loss_fg_3.data - loss_fg_1.data))
        # m3 = torch.mean(loss_fg_3.data)
        # n3 = torch.mean(torch.abs(loss_fg_1.data - loss_fg_2.data))

        # print("loss1: %.4f; loss2: %.4f; loss3: %.4f; dev1: %.4f; dev2: %.4f; dev3: %.4f;"%(m1, m2, m3, n1, n2, n3))

        mu_1 = loss_fg_1.data + 3*torch.abs(loss_fg_2.data - loss_fg_3.data)
        mu_2 = loss_fg_2.data + 3*torch.abs(loss_fg_3.data - loss_fg_1.data)
        mu_3 = loss_fg_3.data + 3*torch.abs(loss_fg_1.data - loss_fg_2.data)
        ind_1_sorted = torch.argsort(mu_1)
        ind_2_sorted = torch.argsort(mu_2)
        ind_3_sorted = torch.argsort(mu_3)
        
        num_remember = int(len(ind_1_sorted) * self.ratio)
        loss_1_update = (torch.mean(loss_fg_1[ind_1_sorted[:num_remember]]) + torch.mean(loss_1[mask_bg])) / 2
        loss_2_update = (torch.mean(loss_fg_2[ind_2_sorted[:num_remember]]) + torch.mean(loss_2[mask_bg])) / 2
        loss_3_update = (torch.mean(loss_fg_3[ind_3_sorted[:num_remember]]) + torch.mean(loss_3[mask_bg])) / 2

        if epoch < self.ss_epoch:
            return loss_1_update, loss_2_update, loss_3_update
        else:
            weight = epoch/120*(1-self.w) + self.w

            preds1 = torch.softmax(preds1, dim=1)
            preds2 = torch.softmax(preds2, dim=1)
            preds3 = torch.softmax(preds3, dim=1)
            preds = (preds1 + preds2 + preds3) / 3
            preds = torch.pow(preds, 1/self.T)
            pseudo = preds / torch.sum(preds, dim=1, keepdim=True)
            pseudo = pseudo.detach()
            
            loss_1_pseudo = self.ps_ce(preds1, pseudo)
            loss_2_pseudo = self.ps_ce(preds2, pseudo)
            loss_3_pseudo = self.ps_ce(preds3, pseudo)

            print("loss1_pseudo: %.4f; loss2_pseudo: %.4f; loss3_pseudo: %.4f; loss1_update: %.4f; loss2_update: %.4f; loss3_update: %.4f;"%(
                loss_1_pseudo.data, loss_2_pseudo.data, loss_3_pseudo.data, loss_1_update.data, loss_2_update.data, loss_3_update.data))

            loss_1 = weight * loss_1_pseudo +  (1-weight) * loss_1_update 
            loss_2 = weight * loss_2_pseudo +  (1-weight) * loss_2_update
            loss_3 = weight * loss_3_pseudo +  (1-weight) * loss_3_update

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
