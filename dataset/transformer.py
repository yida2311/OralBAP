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


class Transformer:
    def __init__(self, crop_size=800):
        self.master = albumentations.Compose([
            albumentations.RandomCrop(crop_size, crop_size),
            albumentations.Transpose(p=0.5),
            albumentations.Flip(p=0.5),
            albumentations.OneOf([
                albumentations.RandomBrightness(),
                albumentations.RandomContrast(),
                albumentations.HueSaturationValue(),
            ], p=0.2),
            albumentations.Normalize(mean=MEAN, std=STD),
        ])
        self.to_tensor = ToTensor()

    def __call__(self, image=None, mask=None):
        result = self.master(image=image, mask=mask)
        result['image'] = self.to_tensor(image=result['image'])['image']
        result['mask'] = torch.tensor(result['mask'], dtype=torch.long)
        return result
    

class TransformerVal:
    def __init__(self):
        self.master = albumentations.Compose([
            albumentations.Normalize(mean=MEAN, std=STD),
            ToTensor(),
        ])
    
    def __call__(self, image=None, mask=None):
        result = self.master(image=image)
        if mask is not None:
            result['mask'] = torch.tensor(mask, dtype=torch.long)
        return result




