from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


class CoTeachingLoss(nn.Module):
    def __init__(self):
        super(CoTeachingLoss, self).__init__()

        self.loss_1 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.loss_2 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    
    def forward(self, preds1, preds2, target, forget_rate):
        loss_1 = self.loss_1(preds1, target).view(-1) # NHW
        ind_1_sorted = torch.argsort(loss_1.data)
        loss_2 = self.loss_2(preds2, target).view(-1)
        ind_2_sorted = torch.argsort(loss_2.data)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(ind_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]

        loss_1_update = torch.sum(loss_1[ind_2_update]) / num_remember
        loss_2_update = torch.sum(loss_2[ind_1_update]) / num_remember

        return loss_1_update, loss_2_update