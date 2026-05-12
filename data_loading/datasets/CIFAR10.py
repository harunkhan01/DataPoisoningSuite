from torchvision import datasets

from ..registry import register_dataset
from ..base import GenericDataset

@register_dataset("cifar10")
class CIFAR10(GenericDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build(self):
        self.train = self.build_cifar10(train=True, transform=self.transform) 
        self.test  = self.build_cifar10(train=False, transform=self.transform)

    def build_cifar10(self, train, transform):

        dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)

        return dataset