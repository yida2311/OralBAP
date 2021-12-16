from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


class BapTALoss(nn.Module):
    """
    Hybrid Loss for BapnetTA: 
        Segmentation Loss: gt seg loss + sim-pseudo seg loss 
        Classification Loss: cls loss
        Consistency Loss: seg consistency loss + sim consisitency loss + seg-sim consistency
        (potential) Size Const Loss: log-barrier
    """
    def __init__(self,
                alpha = 1.0, # for cls loss
                beta = 1.0, # for size const loss
                gamma = 1.0, # for cons loss
                delta = 1.0, # for sim loss
                w = 0.5,  # for curriculum learning
                use_size_const = False,
                use_cons_loss = True,
                use_sim_loss = False,
                use_curriculum = False,
                use_sim_weight = True,
                cons_type = 'mse', # 
                aux_params: Optional[dict] = None,
                ):
        super(BapTALoss, self).__init__()
        self.gt_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if use_sim_weight:
            self.seg_loss = _WeightedCELoss()
        else:
            self.seg_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if use_size_const:
            self.elb_loss = _ExtendedLBLoss(**aux_params)  # Extended Log-Barrier Loss
        if use_cons_loss:
            if cons_type == 'mse':
                self.cons_loss = nn.MSELoss(reduction='mean')
            else:
                self.cons_loss = kl_loss
        if use_sim_loss:
            if cons_type == "mse":
                self.sim_loss = nn.MSELoss(reduction='mean')
            else:
                self.sim_loss = kl_loss

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.use_size_const = use_size_const
        self.use_cons_loss = use_cons_loss
        self.use_sim_loss = use_sim_loss
        self.use_curriculum = use_curriculum
        self.use_sim_weight = use_sim_weight

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
        loss_bg = self.elb_loss(-l1_bg)
        # foreground
        l1_fg = torch.abs(mask_pred.contiguous().view(bs, -1)).sum(dim=1)
        l1_fg = l1_fg / float(h*w)
        loss_fg = self.elb_loss(-l1_fg)

        loss = loss_bg + loss_fg
        return loss
    
    def forward(self, seg_feat, seg_label, cls_feat, cls_label, sim_q, sim_k, gt_label, epoch):
        loss_term = dict()
        if self.use_sim_weight:
            weight = sim_q.clone().detach()
            weight[gt_label==1] = 1
            weight[seg_label!=1] = 1 - weight[seg_label!=1]
            seg_term = self.seg_loss(seg_feat, seg_label, weight)
        else:
            seg_term = self.seg_loss(seg_feat, seg_label)
        loss_term["pseudo_seg_loss"] = seg_term.item()
        loss = seg_term

        if self.use_curriculum:
            gt_term = self.gt_loss(seg_feat, gt_label)
            loss_term["gt_seg_loss"] = gt_term.item()
            w = epoch/self.T*(1-self.w) + self.w
            loss =  w* loss + (1-w) * gt_term 

        cls_term = self.cls_loss(cls_feat, cls_label)
        loss_term["cls_loss"] = cls_term.item()
        loss = loss + self.alpha * cls_term

        if self.use_size_const:
            size_term = self.size_const(1-sim_q)
            loss_term["size_loss"] = size_term.item()
            loss += self.beta * size_term

        if self.use_cons_loss:
            cons_term = self.cons_loss(sim_q, sim_k)
            loss_term["cons_loss"] = cons_term.item()
            loss += self.gamma * cons_term
        
        if self.use_sim_loss:
            target = (seg_label==1).clone().detach().float()
            sim_term = self.sim_loss(sim_q, target) 
            loss_term["sim_loss"] = sim_term.item()
            loss += self.delta * sim_term
        
        return loss, loss_term


class _WeightedCELoss(nn.Module):
    def __init__(self):
        super(_WeightedCELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        
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


def kl_loss(input_logits, target_logits):
    """input_logits: N x H x W"""
    assert input_logits.size() == target_logits.size()
    input_log = torch.log(torch.clamp(input_logits, min=1e-4))
    input_fg_log = torch.log(torch.clamp(1-input_logits, min=1e-4))
    target = torch.clamp(target_logits, min=1e-4)
    target_fg = torch.clamp(1-target_logits, min=1e-4)
    loss = F.kl_div(input_log, target, size_average=True) + F.kl_div(input_fg_log, target_fg, size_average=True)
    return loss / 2




