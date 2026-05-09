import torch.datasets as datasets

from ..registry import register_dataset

@register_dataset("cifar10")
class CIFAR10:
    def __init__(self, cfg):
        pass