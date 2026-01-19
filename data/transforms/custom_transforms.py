# data/transforms/custom_transforms.py
import torch

from data.transforms.registry import register_transform


@register_transform("AddGaussianNoise")
class AddGaussianNoise:
    """Add Gaussian noise to a tensor image."""

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if not torch.is_tensor(img):
            raise TypeError(f"Expected torch.Tensor but got {type(img)}")
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise

