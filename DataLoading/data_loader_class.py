"""
Implement a data loader class that is capable of
    - Retrieving particular datasets
    - Loading subsets of datasets
"""
import torch
import torchvision.datasets as datasets

class DataLoaderClass:
    def __init__(self, dataset_name, preprocess, batch_size=64, split=[0.8, 0.2]):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.transform = preprocess
        self.train_val_split = split

        self.train_ds = None
        self.test_ds = None
        self.load_dataset()
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None

    def load_dataset(self):
        if "mnist" in self.dataset_name:
            self.train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
            self.test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        elif "cifar100" in self.dataset_name:
            pass
        elif "cifar10" in self.dataset_name:
            pass
        elif "svhn" in self.dataset_name:
            pass
        elif "imagenette" in self.dataset_name:
            pass

    def dataloader_exists(self):
        return True if self.train_dl is not None else False

    def construct_dataloader(self):
        pass