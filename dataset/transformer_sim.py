import cv2
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import random
import albumentations
from albumentations.pytorch import ToTensor
from PIL import Image, ImageFile

from .utils import np2pil

ImageFile.LOAD_TRUNCATED_IMAGES = True
MEAN = [0.798, 0.621, 0.841]
STD = [0.125, 0.228, 0.089]


def get_random_crop_coords(height: int, width: int, crop_height: int, crop_width: int, h_start: float, w_start: float):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2

def transpose(img):
    return img.transpose(1,0,2) if len(img.shape) > 2 else img.transpose(1, 0)


class RandomCropSim:
    def __init__(self, crop_size=800, p=1):
        self.crop_height, self.crop_width = crop_size, crop_size
        self.p = p
    
    def __call__(self, image=None, masks=None):
        height, width = image.shape[:2]
        assert height >= self.crop_height and width >= self.crop_width
        
        x1, y1, x2, y2 = 0, self.crop_width, 0, self.crop_height
        if random.random() < self.p:
            h_start, w_start = random.random(), random.random()
            x1, y1, x2, y2 = get_random_crop_coords(height, width, self.crop_height, self.crop_width, h_start, w_start)
            image = image[y1:y2, x1:x2]
            masks = [mask[y1:y2, x1:x2] for mask in masks]
        coord = np.array([x1, y1, x2, y2], dtype=np.int)
        return image, masks, coord

class RandomFlipSim:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image=None, masks=None):
        d = 255
        if random.random() < self.p:
            d = random.randint(-1, 1)
            image = cv2.flip(image, d)
            masks = [cv2.flip(mask, d) for mask in masks]
        
        return image, masks, d

class RandomTransposeSim:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image=None, masks=None):
        d = 0
        if random.random() < self.p:
            image = transpose(image)
            masks = [transpose(mask) for mask in masks]
            d = 1
        
        return image, masks, d


class TransformerSim:
    def __init__(self, crop_size=800):
        self.random_crop = RandomCropSim(crop_size=crop_size)
        self.transpose =RandomTransposeSim(p=0.5)
        self.flip =RandomFlipSim(p=0.5)
        self.master = albumentations.Compose([
            albumentations.OneOf([
                albumentations.RandomBrightness(),
                albumentations.RandomContrast(),
                albumentations.HueSaturationValue(),
            ], p=0.2),
            albumentations.Normalize(mean=MEAN, std=STD),
        ])
        self.to_tensor = ToTensor()

    def __call__(self, image=None, masks=None):
        image, masks, coord = self.random_crop(image=image, masks=masks)
        image, masks, trans_code = self.transpose(image=image, masks=masks)
        image, masks, flip_code = self.flip(image=image, masks=masks)
        image = self.master(image=image)['image']

        image = self.to_tensor(image=image)['image']
        sim = self.to_tensor(image=masks[1])['image']
        mask = torch.tensor(masks[0], dtype=torch.long)
        result = dict(image=image, sim=sim, mask=mask, crop_param=coord, transpose_param=trans_code, flip_param=flip_code)
        return result


def inverseTransformerSim(sim, old_sim, crop_param=None, transpose_param=None, flip_param=None):
    if flip_param:
        if flip_param != 255:
            sim = cv2.flip(sim, flip_param)
    if transpose_param:
        if transpose_param == 1:
            sim = transpose(sim)
    if crop_param is not None:
        x1, y1, x2, y2 = crop_param
        old_sim[y1:y2, x1:x2] = sim
    else:
        old_sim = sim 
    
    return old_sim

