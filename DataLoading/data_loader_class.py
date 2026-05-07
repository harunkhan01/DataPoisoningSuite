"""
Implement a data loader class that is capable of
    - Retrieving particular datasets
    - Loading subsets of datasets
"""
import torch
import torchvision.datasets as datasets

class DataLoaderClass:
    def __init__(self, dataset_name, batch_size=64, split=[0.8, 0.2]):
        self.dataset_name = dataset_name
        self.train_test_split = split
        self.dataset = None
        self.load_dataset()

    def load_dataset(self):
        if "mnist" in self.dataset_name:
            pass
        elif "cifar100" in self.dataset_name:
            pass
        elif "cifar10" in self.dataset_name:
            pass
        elif "svhn" in self.dataset_name:
            pass
        elif "imagenette" in self.dataset_name:
            pass