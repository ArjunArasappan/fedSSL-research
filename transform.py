import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms
import PIL

import random



class SimCLRTransform:
    def __init__(self, size=32):
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.size = size

        self.augment_tranaform = [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        
        if random.choice([True, False]):
            self.augment_transform.append(torchvision.transforms.GaussianBlur(int(0.1 * size), sigma = random.uniform(0.1, 2)))

                
        self.augment_tranaform.append(transforms.ToTensor())

    def __call__(self, x):

        transform = transforms.Compose(self.augment_transform)
        no_augment = self.test_transform()

        return (no_augment(x), transform(x), transform(x))

    def test_transform(self):
        return transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])
