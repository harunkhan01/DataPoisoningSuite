import torch.datasets as datasets

from ..registry import register_dataset

@register_dataset("cifar100")
class CIFAR100:
    def __init__(self, cfg):
        pass