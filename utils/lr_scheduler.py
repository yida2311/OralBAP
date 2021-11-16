##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
from torch.optim.lr_scheduler import _LRScheduler

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    YM mode

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0.0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step  # for mode: step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch  # overall iterations
        self.warmup_iters = int(warmup_epochs * iters_per_epoch)
        self.thr = [10, 60]  # for mode: ym

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i # iter index
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        elif self.mode == 'ym':
            scale = 1
            if epoch >= self.thr[0]:
                scale = 0.1 * scale
                if epoch >= self.thr[1]:
                    scale = scale * (0.9 ** (epoch-self.thr[1]))
            lr = self.lr * scale
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

        return lr

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            for i in range(len(optimizer.param_groups)):
                if optimizer.param_groups[i]['lr'] > 0: 
                    optimizer.param_groups[i]['lr'] = lr
            

class CycleScheduler(_LRScheduler):
    """ Consine annealing with warm up and restarts.
    Proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts`.
    """
    def __init__(self, 
                iters_per_epoch=0,
                total_epoch=100, 
                # start_cyclical=10, 
                cyclical_base_lr=7e-4, 
                cyclical_epoch=10, 
                eta_min=0, 
                warmup_epoch=5, 
                last_epoch=-1):
        self.total_epoch = total_epoch
        # self.start_cyclical = start_cyclical
        self.cyclical_epoch = cyclical_epoch
        self.cyclical_base_lr = cyclical_base_lr
        self.eta_min = eta_min
        self.warmup_epoch = warmup_epoch
        self.iters_per_epoch = iters_per_epoch

        self.warmup_iters = int(self.warmup_epoch * self.iters_per_epoch)
        self.cycle_iters = int(self.cyclical_epoch * self.iters_per_epoch)


    def __call__(self, optimizer, i, epoch):
        if epoch < self.warmup_epoch:
            T = epoch * self.iters_per_epoch + i # iter index
            lr = self.cyclical_base_lr * 1.0 * T / self.warmup_iters
        else:
            T = ((epoch - self.warmup_epoch) % self.cyclical_epoch) * self.iters_per_epoch + i
            lr = self.eta_min + 0.5 * (self.cyclical_base_lr - self.eta_min) * (1 + math.cos(T / self.cycle_iters * math.pi))

        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            for i in range(len(optimizer.param_groups)):
                if optimizer.param_groups[i]['lr'] > 0: 
                    optimizer.param_groups[i]['lr'] = lr











