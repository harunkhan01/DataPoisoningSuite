import torch
import random
from torch.utils.data import DataLoader

class EvaluatePoisons:
    def __init__(self, model, dataset, poisoned_dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.poisoned_dataset = poisoned_dataset

    def evaluate(self):
        benign_acc = self.evaluate_dataset(self.dataset)

        poison_acc = self.evaluate_dataset(self.poisoned_dataset)

        print(f'Accuracy of model on benign samples : {benign_acc}')

        print(f'Accuracy of model on poison samples : {poison_acc}')

        return benign_acc, poison_acc

    def evaluate_dataset(self, ds):
        self.model.eval()
        eval_dataloader = DataLoader(ds, batch_size=10000, shuffle=False)
        
        correct = 0
        total = len(ds)
        for x, y in eval_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            outputs = self.model(x)
            correct += ((torch.argmax(outputs, dim=1)) == y).sum().item()
            
        return correct / total