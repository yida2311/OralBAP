import os
import random
import torch 
import math
import cv2
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .utils import cv2_image_loader, cv2_mask_loader


class OralDataset(Dataset):
    """ OralDataset for patch segmentation"""
    def __init__(self,
                img_dir,
                mask_dir,
                meta_file, # csv file
                label=True,  # True: has label
                transform=None):
        super(OralDataset, self).__init__()
        self.img_dir = img_dir
        self.label = label 
        if self.label:
            self.mask_dir = mask_dir
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
                sample = self.transform(image=img, mask=mask)
            else:
                sample = self.transform(image=img)
        sample['id'] = info['image_id'].split('.')[0]

        return sample
    
    def __len__(self):
        return len(self.samples)


class OralSlide(Dataset):
    """OralSlide for slide segmentation"""
    def __init__(self,
                slide_list,
                img_dir, 
                mask_dir, 
                slide_file, # meta info for slide
                slide_mask_dir=None,
                label=False,
                transform=None):
        """
        Params:
            slide_list: list of slides name
            img_dir: image directory
            slide_file: json file, slide meta file, which include size, tiles, step
            slide_mask_dir: mask directory
            label: if True, used for train/val; if False, used for test
            transform: image preprocess
        """
        super(OralSlide, self).__init__()
        self.slides = slide_list
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.label = label 
        if self.label:
            self.slide_mask_dir = slide_mask_dir
        self.transform = transform

        with open(slide_file, 'r') as f:
            cnt = json.load(f)
        self.info = cnt  # {"slide": {"size":[h, w], "tiles": [x, y], "step":[step_x, step_y]}}

        self.slide = "" # current slide
        self.slide_size = None
        self.slide_step = None
        self.samples = []

    def get_slide_mask_from_index(self, index):
        """Generate  slide mask based on index from patch mask
            ---no use for test 
        """
        slide = self.slides[index]
        slide_mask_dir = os.path.join(self.slide_mask_dir, slide+'.png')
        slide_mask = cv2_mask_loader(slide_mask_dir)
        
        return slide_mask

    def get_patches_from_index(self, index):
        """Collect slide info and patches based on index"""
        self.slide = self.slides[index]
        slide_dir = os.path.join(self.img_dir, self.slide)
        self.samples = os.listdir(slide_dir)

        size = self.info[self.slide]['size']
        step = self.info[self.slide]['step']
        self.slide_size = tuple(size)
        self.slide_step = step
    
    def __getitem__(self, index):
        patch = self.samples[index]
        img_path = os.path.join(os.path.join(self.img_dir, self.slide), patch)
        img = cv2_image_loader(img_path)
        sample = {}

        if self.transform:
            if self.label:
                mask_path = os.path.join(os.path.join(self.mask_dir, self.slide), patch)
                mask = cv2_mask_loader(mask_path)
                sample = self.transform(image=img, mask=mask)
            else:
                sample = self.transform(image=img)
        
        ver, col = self._parse_patch_name(patch)
        sample['coord'] = (ver, col)
        sample['id'] = patch.split('.')[0]
        return sample
    
    def __len__(self):
        return len(self.samples)

    def _parse_patch_name(self, patch):
        sp = patch.split('_')
        ver = int(sp[1])
        col = int(sp[2])
        return ver, col 








