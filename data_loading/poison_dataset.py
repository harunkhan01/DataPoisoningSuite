import torch
import random
from torch.utils.data import Dataset

class PoisonedDataset(Dataset):
    def __init__(self, dataset, poison_obj, cfg):
        self.dataset = dataset
        self.poison_obj = poison_obj
        self.target_label = cfg['target_label']

        n = len(self.dataset)
        n_poison = int(n * cfg['poison_ratio'])
        self.poisoned_indices = set(
            random.sample(range(n), n_poison)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        x, y = self.dataset[idx]
        
        if idx in self.poisoned_indices:
            x = self.poison_obj.apply(x)
            y = self.target_label

        return x, y 