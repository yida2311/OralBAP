from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional

__all__ = ['SegClsLoss', 'SegClsLoss_v2', 'CrossEntropyLoss', 'SymmetricCrossEntropyLoss', 'NormalizedSymmetricCrossEntropyLoss', 
            'FocalLoss', 'SoftCrossEntropyLoss2d']


#===================Seg Cls Loss ==============================
class SegClsLoss(nn.Module):
    def __init__(self,
                alpha = 1.0,
                beta = 1e-2,
                use_size_const = False,
                use_curriculum = False,
                aux_params: Optional[dict] = None,
                ):
        super(SegClsLoss, self).__init__()
        self.gt_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.seg_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.cls_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if use_size_const:
            self.elb_loss = _ExtendedLBLoss(**aux_params)  # Extended Log-Barrier Loss

        self.alpha = alpha
        self.beta = beta
        self.use_size_const = use_size_const
        self.use_curriculum = use_curriculum
        self.T = 120
    
    def size_const(self, mask_pred):
        """"
        Compute the loss over the size of the mask.
        :param mask_pred: foreground predicted mask, shape: (n, 1, h, w)
        """
        assert mask_pred.ndim == 4

        # background
        bgmask = 1 - mask_pred
        bs, _, h, w = mask_pred.size()
        l1_bg = torch.abs(bgmask.contiguous().view(bs, -1)).sum(dim=1)
        l1_bg = l1_bg / float(h*w)
        loss_bg = self.elb_loss(-l1_bg)

        l1_fg = torch.abs(mask_pred.contiguous().view(bs, -1)).sum(dim=1)
        l1_fg = l1_fg / float(h*w)
        loss_fg = self.elb_loss(-l1_fg)

        loss = loss_bg + loss_fg

        return loss

    
    def forward(self, seg_feat, seg_label, cls_feat, cls_label, sim, gt_label, epoch):
        seg_term = self.seg_loss(seg_feat, seg_label)
        loss = seg_term
        if self.use_curriculum:
            w = epoch/self.T*0.8 + 0.2
            gt_term = self.gt_loss(seg_feat, gt_label)
            loss = (1-w) * gt_term + w * loss

        cls_term = self.cls_loss(cls_feat, cls_label)
        loss = loss + self.alpha * cls_term

        if self.use_size_const:
            elb_term = self.elb_loss(seg_feat)
            loss += self.beta * elb_term
        
        return loss


class SegClsLoss_v2(nn.Module):
    def __init__(self,
                alpha = 1.0,
                beta = 1e-2,
                use_size_const = False,
                use_curriculum = False,
                aux_params: Optional[dict] = None,
                ):
        super(SegClsLoss_v2, self).__init__()
        self.gt_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.seg_loss = _WeightedCELoss()
        self.cls_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if use_size_const:
            self.elb_loss = _ExtendedLBLoss(**aux_params)  # Extended Log-Barrier Loss

        self.alpha = alpha
        self.beta = beta
        self.use_size_const = use_size_const
        self.use_curriculum = use_curriculum
        self.T = 120 
    
    def size_const(self, mask_pred):
        """"
        Compute the loss over the size of the mask.
        :param mask_pred: foreground predicted mask, shape: (n, 1, h, w)
        """
        assert mask_pred.ndim == 4

        # background
        bgmask = 1 - mask_pred
        bs, _, h, w = mask_pred.size()
        l1_bg = torch.abs(bgmask.contiguous().view(bs, -1)).sum(dim=1)
        l1_bg = l1_bg / float(h*w)
        loss_bg = self.elb_loss(-l1_bg)

        l1_fg = torch.abs(mask_pred.contiguous().view(bs, -1)).sum(dim=1)
        l1_fg = l1_fg / float(h*w)
        loss_fg = self.elb_loss(-l1_fg)

        loss = loss_bg + loss_fg

        return loss

    
    def forward(self, seg_feat, seg_label, cls_feat, cls_label, sim, gt_label, epoch):
        weight = sim.clone().detach()
        weight[gt_label==1] = 1
        weight[seg_label!=1] = 1 - weight[seg_label!=1]
        seg_term = self.seg_loss(seg_feat, seg_label, weight)
        loss = seg_term
        if self.use_curriculum:
            gt_term = self.gt_loss(seg_feat, gt_label)
            w = epoch/self.T*0.8 + 0.1
            loss =  w* loss + (1-w) * gt_term

        cls_term = self.cls_loss(cls_feat, cls_label)
        loss = loss + self.alpha * cls_term

        if self.use_size_const:
            elb_term = self.elb_loss(seg_feat)
            loss += self.beta * elb_term
        
        return loss


class _WeightedCELoss(nn.Module):
    def __init__(self):
        super(_WeightedCELoss, self).__init__()
        self.ce = CrossEntropyLoss(ignore_index=-1, reduction='none')
    
    def forward(self, input, label, weight):
        loss = self.ce(input, label)
        loss = torch.mean(loss*weight, dim=(0,1,2))

        return loss


class _ExtendedLBLoss(nn.Module):
    """
    Extended Log-Barrier loss (ELB).
    Optimize inequality constraint: f(x) <= 0.
    """
    def __init__(self, init_t=1, max_t=10, mulcoef=1.01):
        """
        :param init_t: float > 0. The initial value of t.
        :param max_t: float >0. The maximum value of t.
        :param mulcoef: float >0. The coefficient use to update t in the form: t = t * mulcoef.
        """
        super(_ExtendedLBLoss, self).__init__()
        self.init_t = init_t
        self.register_buffer(
            "mulcoef", torch.tensor([mulcoef], requires_grad=False).float()
        )
        self.register_buffer(
            "t_lb", torch.tensor([init_t], requires_grad=False).float()
        )
        self.register_buffer(
            "max_t", torch.tensor([max_t], requires_grad=False).float()
        )
    
    def set_t(self, val):
        if isinstance(val, float):
            self.register_buffer(
                "t_lb", torch.tensor([val], requires_grad=False).float().to(self.t_lb.device)
            )
        elif isinstance(val, torch.Tensor):
            self.register_buffer(
                "t_lb", val.float().requires_grad(False)
            )

    def get_t(self):
        return self.t_lb
    
    def update_t(self):
        self.set_t(torch.min(self.t_lb * self.mul_coef), self.max_t)
    
    def forward(self, fx):
        assert fx.ndim == 1, "fx.ndim must be 1. found {}".format(fx.ndim)

        loss_fx = fx * 0

        # vals <= -1/(self.t_lb**2)
        ct = - (1. / (self.t_lb**2))

        idx_less = (fx <= ct).nonzero().squeeze()
        if idx_less.numel() > 0:
            val_less = fx[idx_less]
            loss_less = -(1. / self.t_lb) * torch.log( - val_less)
            loss_fx[idx_less] = loss_less
        
        idx_great = (fx > ct).nonzero().squeeze()
        if idx_great.numel() > 0:
            val_great = fx[idx_great]
            loss_great = self.t_lb * val_great -(1. / self.t_lb) * torch.log(1. / (self.t_lb**2)) + 1. / self.t_lb
            loss_fx[idx_great] = loss_great
        
        loss = loss_fx.mean()

        return loss


#================== KL Divergence Loss =========================
class KLDivergenceLoss(nn.Module):
    def __init__(self, T=1):
        super(KLDivergenceLoss, self).__init__()
        self.T = T

    def forward(self, input, target):
        log_input_prob = F.log_softmax(input / self.T, dim=1)
        target_porb = F.softmax(target / self.T, dim=1)
        loss = F.kl_div(log_input_prob, target_porb)
        return self.T*self.T*loss # balanced


#================== Symmetric Cross Entropy Loss =========================
class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=4, ignore_index=-100, reduction='mean'):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.rce_loss = ReverseCrossEntropyLoss(num_classes=num_classes, reduction=reduction)

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        rce = self.rce_loss(inputs, targets)

        loss = self.alpha * ce + self.beta * rce 
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class ReverseCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4, ignore_index=-100, reduction='mean'):
        super(ReverseCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        pred = F.softmax(inputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        if inputs.dim() == 4:
            B, C, H, W = inputs.size()
            pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
            targets = targets.view(-1)

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            pred = pred[valid]
            targets = targets[valid]

        targets_one_hot = one_hot(targets, self.num_classes)
        targets_one_hot = torch.clamp(targets_one_hot, min=1e-4, max=1.0)

        loss = -1 * torch.sum(pred * torch.log(targets_one_hot), dim=1)
        # print(loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


#===================== Normalized Symmetric Cross Entropy Loss =====================
class NormalizedSymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=4, ignore_index=-100, reduction='mean'):
        super(NormalizedSymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nce_loss = NormalizedCrossEntropyLoss(num_classes=num_classes, ignore_index=ignore_index, reduction=reduction)
        self.nrce_loss = NormalizedReverseCrossEntropyLoss(num_classes=num_classes, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, targets):
        nce = self.nce_loss(inputs, targets)
        nrce = self.nrce_loss(inputs, targets)
        
        loss = self.alpha * nce + self.beta * nrce 
        return loss


class NormalizedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4, ignore_index=-100, reduction='mean'):
        super(NormalizedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        pred = F.log_softmax(inputs, dim=1)
        B, C, H, W = inputs.size()
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        targets = targets.view(-1)

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            pred = pred[valid]
            targets = targets[valid]

        targets_one_hot = one_hot(targets, self.num_classes)
        ce =  -1 * torch.sum(targets_one_hot * pred, dim=1)
        C = -1 * torch.sum(pred, dim=1)
        nce = torch.div(ce, C)

        if self.reduction == 'mean':
            nce = nce.mean()
        elif self.reduction == 'sum':
            nce = nce.sum()

        return nce


class NormalizedReverseCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4, ignore_index=-100, reduction='mean'):
        super(NormalizedReverseCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        pred = F.softmax(inputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        B, C, H, W = inputs.size()
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        targets = targets.view(-1)

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            pred = pred[valid]
            targets = targets[valid]

        targets_one_hot = one_hot(targets, self.num_classes)
        targets_one_hot = torch.clamp(targets_one_hot, min=1e-4, max=1.0)

        rce = -1 * torch.sum(pred * torch.log(targets_one_hot), dim=1)
        nrce = rce / (self.num_classes-1) / 4

        if self.reduction == 'mean':
            nrce = nrce.mean()
        elif self.reduction == 'sum':
            nrce = nrce.sum()

        return nrce


#======================================= Focal Loss ======================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, one_hot=True, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        self.one_hot = one_hot
        self.ignore_index = ignore_index

    def forward(self, input, target):
        '''
        only support ignore at 0
        '''
        if input.dim() == 4:
            B, C, H, W = input.size()
            input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
            target = target.view(-1)

        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            input = input[valid]
            target = target[valid]

        if self.one_hot: 
            target = one_hot(target, input.size(1))

        probs = F.softmax(input, dim=1)
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()
        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
        
        return loss


#=============== Soft Cross Entropy Loss ========================
class SoftCrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss2d, self).__init__()

    def forward(self, inputs, targets):
        loss = 0
        inputs = -F.log_softmax(inputs, dim=1)
        for index in range(inputs.size()[0]):
            loss += F.conv2d(inputs[range(index, index+1)], targets[range(index, index+1)])/(targets.size()[2] *
                                                                                             targets.size()[3])
        return loss

#-------------------------------- helper function ----------------------------------#

def moving_average(target1, target2, alpha=1.0):
    target = 0
    target += (1.0 - alpha) * target1
    target += target2 * alpha
    return target


def one_hot(index, classes):
    # index is not flattened (pypass ignore) ############
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]
    #####################################################
    # index is flatten (during ignore) ##################
    # size = index.size()[:1] + (classes,)
    # view = index.size()[:1] + (1,)
    #####################################################

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)





