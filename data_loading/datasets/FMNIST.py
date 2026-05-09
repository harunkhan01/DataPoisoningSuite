import torch.datasets as datasets

from ..registry import register_dataset
from ..base import GenericDataset

@register_dataset("fashion_mnist")
class FashionMNIST(GenericDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build(self):
        self.train = self.build_fashion_mnist(train=True, transform=self.transform) 
        self.test  = self.build_fashion_mnist(train=False, transform=self.transform)

    def build_fashion_mnist(self, train, transform):

        dataset = datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)

        return dataset