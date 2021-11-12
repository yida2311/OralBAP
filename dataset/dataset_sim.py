import os
import random
from albumentations.pytorch.transforms import img_to_tensor
import torch 
import math
import cv2
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 

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
                sim_path = os.path.join(os.path.join(self.mask_dir, info['slide_id']), info['image_id'].split('.')[0]+'.npy')
                sim = np.load(sim_path)
                sample = self.transform(image=img, masks=[mask, sim])
            else:
                sample = self.transform(image=img)
        sample['id'] = info['image_id'].split('.')[0]

        return sample
    
    def __len__(self):
        return len(self.samples)
    


if __name__ == '__main__':
    from transformer_sim import TransformerSim
    from ..configs.config_bapnet import Config

    cfg = Config() 

    dataset = OralDatasetSim(
        cfg.trainset_cfg["img_dir"],
        cfg.trainset_cfg["mask_dir"],
        cfg.trainset_cfg["sim"],
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