import math
import numpy as np
import random
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from einops import rearrange

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask, edge = None,masked=None):
        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size)

        if edge is not None:
            edge = F.resize(edge, self.size)
        if masked is not None:
            masked = F.resize(masked, self.size)
            
        return image, mask, edge, masked
    
class RandomCrop(object):
    
    def __init__(self, size):
        self.size = size
    def __call__(self, image, target, edge = None,masked=None):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)

        if edge is not None:
            edge = F.crop(edge, self.size)
        if masked is not None:
            masked = F.crop(masked, *crop_params)
        return image, target, edge, masked

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
    def __call__(self, image, target, edge = None,masked = None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)

            if edge is not None:
                edge = F.hflip(edge)
            if masked is not None:
                masked = F.hflip(masked)
        return image, target, edge, masked
    
class CenterCrop(object):

    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target, edge = None,masked = None):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)

        if edge is not None:
            edge = F.center_crop(edge, self.size)
        if masked is not None:
            masked = F.center_crop(masked, self.size)
        return image, target, edge, masked
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
 
    def __call__(self, image, target, edge = None,masked = None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        target = target / 255.

        if edge is not None:
            edge = edge / 255.
        if masked is not None:
            masked = F.normalize(masked, mean=self.mean, std=self.std)
        return image, target, edge, masked

class Pad(object):
     

    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value
 
    def __call__(self, image, target, edge = None,masked = None):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        target = F.pad(target, self.padding_n, self.padding_fill_target_value)

        if edge is not None:
            edge = F.pad(edge, self.padding_n, self.padding_fill_target_value)
        if masked is not None:
            masked = F.pad(masked, self.padding_n, self.padding_fill_target_value)
        return image, target, edge, masked
    
    
class ToTensor(object):
    def __call__(self, image, target, edge = None,masked = None):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.float)
        target = rearrange(target,'h w c->c h w')

        if edge is not None:
            edge = torch.as_tensor(np.array(edge), dtype=torch.float)
            edge = rearrange(edge,'h w c->c h w')
        if masked is not None:
            masked = F.to_tensor(masked)
        return image, target, edge, masked


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, mask, edge = None, masked = None):
        for t in self.transforms:
            image, mask, edge, masked = t(image, mask, edge, masked)
        mask = mask[0:1,:,:]
        if edge is not None:
            edge = edge[0:1,:,:]
        return {'image':image, 'mask':mask, 'edge': edge, 'masked':masked}
 
 
