import torch.datasets as datasets

from ..registry import register_dataset
from ..base import GenericDataset

@register_dataset("cifar100")
class CIFAR100(GenericDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build(self):
        self.train = self.build_cifar100(train=True, transform=self.transform) 
        self.test  = self.build_cifar100(train=False, transform=self.transform)

    def build_cifar100(self, train, transform):

        dataset = datasets.CIFAR100(root="./data", train=train, download=True, transform=transform)

        return dataset