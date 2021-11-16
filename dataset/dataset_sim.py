import os
import random
import torch 
import math
import cv2
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from utils.util import simple_time 

from .utils import cv2_image_loader, cv2_mask_loader


class OralDatasetSim(Dataset):
    """ OralDataset for patch segmentation"""
    def __init__(self,
                img_dir,
                mask_dir,
                sim_dir,
                meta_file, # csv file
                label=True,  # True: has label
                transform=None):
        super(OralDatasetSim, self).__init__()
        self.img_dir = img_dir
        self.label = label 
        if self.label:
            self.mask_dir = mask_dir
            self.sim_dir = sim_dir
        self.transform = transform
        self.to_tensor = ToTensor()

        df = pd.read_csv(meta_file)
        self.samples = df  
        # self.ids = list(range(len(self.samples)))
        # random.shuffle(self.ids)
    
    def __getitem__(self, index):
        info = self.samples.iloc[index]
        img_path = os.path.join(os.path.join(self.img_dir, info['slide_id']), info['image_id'])
        img = cv2_image_loader(img_path)
        sample = {}
        
        if self.transform:
            if self.label:
                mask_path = os.path.join(os.path.join(self.mask_dir, info['slide_id']), info['image_id'])
                mask = cv2_mask_loader(mask_path)
                sim_path = os.path.join(os.path.join(self.sim_dir, info['slide_id']), info['image_id'])
                if os.path.exists(sim_path):
                    sim = cv2_image_loader(sim_path)
                else:
                    sim = class_to_sim(mask)
                sample = self.transform(image=img, masks=[mask, sim])
                sample['full_sim'] = self.to_tensor(sim)
            else:
                sample = self.transform(image=img)
        sample['id'] = info['image_id'].split('.')[0]

        return sample
    
    def __len__(self):
        return len(self.samples)
    

CLASS_COLORS = [
    [85, 85 ,85],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
]

def class_to_sim(label):
    h, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(h, w, 3)).astype(np.uint8)
    
    for i in range(4):
        indices = np.where(label == i)
        colmap[indices[0].tolist(), indices[1].tolist(), :] = CLASS_COLORS[i]
    return colmap


if __name__ == '__main__':
    from transformer_sim import TransformerSim
    from ..configs.config_bapnet import Config

    cfg = Config() 

    dataset = OralDatasetSim(
        cfg.trainset_cfg["img_dir"],
        cfg.trainset_cfg["mask_dir"],
        cfg.trainset_cfg["sim_dir"],
        cfg.trainset_cfg["meta_file"], 
        label=cfg.trainset_cfg["label"], 
        transform=TransformerSim(crop_size=cfg.crop_size),
    )
    dataloader = DataLoader(dataset, 
                                num_workers=cfg.num_workers,
                                batch_size=cfg.batch_size,
                                shuffle=True
    )
    
    for i, sample in enumerate(dataloader):
        for k, v in sample.items():
            print(k)
            print(v)

        if i == 5:
            break