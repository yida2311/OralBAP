from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional

# __all__ = ['SegClsLoss', 'SegTALoss', 'SegMTLoss' 'CrossEntropyLoss', 'SymmetricCrossEntropyLoss', 'NormalizedSymmetricCrossEntropyLoss', 
#             'FocalLoss', 'SoftCrossEntropyLoss2d']

#======================BAP MT Loss=============================
class BapMTLoss(nn.Module):
    """
    Hybrid Loss: 
        Segmentation Loss: seg loss + seg pseudo seg loss + sim seg pseudo loss
        Classification Loss: cls loss
        Consistency Loss: seg consistency loss + sim consisitency loss + seg-sim consistency
    """
    def __init__(self,
                alpha = 1.0,  # cls loss weight
                beta = 1,  # cons loss weight
                w = 0.5,
                use_curriculum = True,
                aux_params: Optional[dict] = None,
                ):
        super(BapMTLoss, self).__init__()
        self.seg_loss =  CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.cls_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.cons_seg_loss = softmax_kl_loss
        self.cons_loss = nn.MSELoss(reduction='mean')
    
        self.alpha = alpha
        self.beta = beta

        self.use_curriculum = use_curriculum
        self.T = 120 
        self.w = w 

    def forward(self, gt_label, epoch, **outputs):
        seg_feat_q = outputs["seg_feat_q"]

        w = epoch/self.T*(1-self.w) + self.w
        seg_gt_term = self.seg_loss(seg_feat_q, gt_label)
        seg_sim_term = self.seg_loss(seg_feat_q, outputs["sim_pseudo_label"])
        seg_seg_term = self.seg_loss(seg_feat_q, outputs["seg_pseudo_label"])
        seg_term = (1-w)*seg_gt_term + w*(seg_sim_term+seg_seg_term)/2

        cls_term = self.cls_loss(outputs["cls_feat"], outputs["cls_label"])

        seg_sim_feat = F.softmax(seg_feat_q, dim=1)[:, 1, ...]
        cons_seg_term = self.cons_seg_loss(seg_feat_q, outputs["seg_feat_k"])
        cons_sim_term = self.cons_loss(outputs["sim_q"], outputs["sim_k"])
        cons_seg_sim_term = self.cons_loss(seg_sim_feat, outputs["sim_q"])
        cons_term = (cons_seg_term+cons_sim_term+cons_seg_sim_term) / 3

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

#===================Seg Cls Loss ==============================
class SegClsLoss(nn.Module):
    def __init__(self,
                alpha = 1.0,
                beta = 1e-2,
                gamma = 1.0,
                w = 0.5,
                use_size_const = False,
                use_sim_loss = True,
                use_curriculum = False,
                sim_weight = True,
                aux_params: Optional[dict] = None,
                ):
        super(SegClsLoss, self).__init__()
        assert use_size_const != use_sim_loss
        self.gt_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if sim_weight:
            self.seg_loss = _WeightedCELoss()
        else:
            self.seg_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.cls_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if use_size_const:
            self.elb_loss = _ExtendedLBLoss(**aux_params)  # Extended Log-Barrier Loss
        if use_sim_loss:
            self.sim_loss = nn.MSELoss(reduction='mean')
            self.cons_loss = nn.KLDivLoss(reduction='mean')

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.use_size_const = use_size_const
        self.use_sim_loss = use_sim_loss
        self.use_curriculum = use_curriculum

        self.init_t = aux_params['init_t']
        self.max_t = aux_params['max_t']
        self.sim_weight = sim_weight
        self.epsilon = 0
        self.T = 120 
        self.w = w
    
    def size_const(self, mask_pred):
        """"
        Compute the loss over the size of the mask.
        :param mask_pred: foreground predicted mask, shape: (n, 1, h, w)
        """
        assert mask_pred.ndim == 3

        # background
        bgmask = 1 - mask_pred
        bs, h, w = mask_pred.size()
        l1_bg = torch.abs(bgmask.contiguous().view(bs, -1)).sum(dim=1)
        l1_bg = l1_bg / float(h*w) 
        l1_bg = l1_bg - self.epsilon
        loss_bg = self.elb_loss(-l1_bg)
        # foreground
        l1_fg = torch.abs(mask_pred.contiguous().view(bs, -1)).sum(dim=1)
        l1_fg = l1_fg / float(h*w)
        l1_fg = l1_fg - self.epsilon
        loss_fg = self.elb_loss(-l1_fg)

        loss = loss_bg + loss_fg

        return loss

    
    def forward(self, seg_feat, seg_label, cls_feat, cls_label, sim_q, sim_k, gt_label, epoch):
        
        if self.sim_weight:
            weight = sim_q.clone().detach()
            weight[gt_label==1] = 1
            weight[seg_label!=1] = 1 - weight[seg_label!=1]
            seg_term = self.seg_loss(seg_feat, seg_label, weight)
        else:
            seg_term = self.seg_loss(seg_feat, seg_label)
        # print(seg_term)
        loss = seg_term

        if self.use_curriculum:
            gt_term = self.gt_loss(seg_feat, gt_label)
            # print(gt_term)
            w = epoch/self.T*(1-self.w) + self.w
            loss =  w* loss + (1-w) * gt_term

        cls_term = self.cls_loss(cls_feat, cls_label)
        # print(cls_term)
        loss = loss + self.alpha * cls_term

        if self.use_size_const:
            elb_term = self.size_const(1-sim_q)
            # print(elb_term)
            loss += self.beta * elb_term

        if self.use_sim_loss:
            target = torch.tensor(seg_label==1, dtype=torch.float)
            sim_term = self.sim_loss(sim_q, target) 
            cons_term = self.cons_loss(sim_q, sim_k)
            # print(sim_term)
            # print(cons_term)
            loss += self.gamma * (sim_term + cons_term)
        
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

#======================    =============================
class SegMTLoss(nn.Module):
    def __init__(self,
                alpha = 1.0,
                w = 0.5,
                use_curriculum = False,
                use_kl = True,
                aux_params: Optional[dict] = None,
                ):
        super(SegMTLoss, self).__init__()
        self.gt_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if use_kl:
            self.cons_loss = softmax_kl_loss
        else:
            self.cons_loss = softmax_mse_loss
        self.seg_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
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
    def __init__(self,
                alpha = 1.0,
                w = 0.5,
                use_curriculum = False,
                aux_params: Optional[dict] = None,
                ):
        super(SegTALoss, self).__init__()
        self.gt_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.seg_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
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





