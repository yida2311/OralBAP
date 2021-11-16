import cv2
import torch
from PIL import Image


def cv2_image_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 

def cv2_mask_loader(path):
    mask = cv2.imread(path, 0)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask

#=====================================================================

def mask_class_merge(mask):
    return torch.clamp(mask, max=2)


def np2pil(np_arr, mode="RGB"):
    return Image.fromarray(np_arr, mode=mode)

#=====================================================================

def collate(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [b[key] for b in batch]
    batch_dict['image'] = torch.stack(batch_dict['image'], dim=0)
    if 'mask' in batch_dict.keys():
        batch_dict['mask'] = torch.stack(batch_dict['mask'], dim=0)

    return batch_dict


def collateGL(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [b[key] for b in batch]
    
    return batch_dict