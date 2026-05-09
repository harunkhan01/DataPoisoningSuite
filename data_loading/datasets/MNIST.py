import torch.datasets as datasets

from ..registry import register_dataset
from ..base import GenericDataset

@register_dataset("mnist")
class MNIST(GenericDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build(self):
        self.train = self.build_mnist(train=True, transform=self.transform) 
        self.test  = self.build_mnist(train=False, transform=self.transform)

    def build_mnist(self, train, transform):

        dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transform)

        return dataset