import torch
import os
from collections import OrderedDict
from torch import nn
import numpy as np 



def model_load_state_dict(model, ckpt_path=None, distributed=False):
    state_dict = torch.load(ckpt_path)
    if 'module' in next(iter(state_dict)) and not distributed:
        state_dict = state_dict_Parallel2Single(state_dict)
    state = model.state_dict()
    state.update(state_dict)
    model.load_state_dict(state)

    return model 

def state_dict_Parallel2Single(original_state):
    converted = OrderedDict()
    for k, v in original_state.items():
        name = k[7:]
        converted[name] = v
    return converted

#=============================================================================

def model_Single2Parallel(model, device, local_rank):
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    return model 

#================================================================================


def save_ckpt_model(model, cfg, iou_fine, best_pred_fine, best_epoch, epoch):
    # save_path = os.path.join(cfg.model_path, "%s-%d-%.4f-%.4f.pth"%(cfg.model+'-'+cfg.encoder, epoch, iou_coarse, iou_fine))
    # torch.save(model.state_dict(), save_path)
    if iou_fine > best_pred_fine:
        best_pred_fine = iou_fine
        best_epoch = epoch
        save_path = os.path.join(cfg.model_path, "%s-best-fine.pth"%(cfg.model+'-'+cfg.encoder))
        torch.save(model.state_dict(), save_path)
        print('Best fine model at epoch %d with mIoU = %.4f'%(epoch, best_pred_fine))

    return best_pred_fine, best_epoch