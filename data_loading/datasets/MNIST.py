import torch.datasets as datasets

from ..registry import register_dataset

@register_dataset("mnist")
class MNIST:
    def __init__(self, cfg):
        pass