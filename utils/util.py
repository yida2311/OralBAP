import random
import os
import numpy as np
import torch
import argparse
import time



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True


def argParser():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    return args


def update_writer(writer, writer_info, epoch):
    for k, v in writer_info.items():
        if isinstance(v, dict):
            writer.add_scalars(k, v, epoch)
        elif isinstance(v, torch.Tensor):
            writer.add_image(k, v, epoch)
        else:
            writer.add_scalar(k, v, epoch)

def struct_time():
    # 格式化成2020-08-07 16:56:32
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return cur_time

def simple_time():
    cur_time = time.strftime("[%m-%d]", time.localtime())
    return cur_time