import torch
import deque
from torch.utils.data import DataLoader

class FinetuneClass:
    def __init__(self, device, cfg):
        self.device = device
        self.cfg = cfg

    def setup_training(self):
        cfg = self.cfg['training']
        self.lr = cfg.get('lr', 0.001)
        self.epochs = cfg.get('epochs', 50)
        self.verbosity = cfg.get('verbosity', 10)
        self.batch_size = cfg.get('batch_size', 64)

    def setup_backbone(self, cfg, model):
        """
        We rely on the fact that model_loading will return a model object with arguments
        backbone and head. These can then be explicitly referred to without searching for 
        them here.
        """
        mode = cfg['mode']

        if mode == 'last_n':
            n = cfg['value']
            modules = list(model.named_modules())

            # initially freeze the backbone
            for param in model.parameters():
                param.requires_grad = False
            
            # set the parameters to be true for last_n
            for _, module in modules[-n:]:
                for param in module.parameters():
                    param.requires_grad = True

        else:
            pass

    def setup_head(self, cfg, model):
        value = cfg['value']

        # freeze if value is 0
        if value == 0:
            modules = list(model.named_modules())

    def setup_model(self, model):
        if 'backbone' in self.cfg:
            self.setup_backbone(self.cfg['backbone'], model)
            
        if 'head' in self.cfg:
            self.setup_head(self.cfg['head'], model)

    def train(self, model, train, test):
        
        self.setup_training()
        self.setup_model(model)

        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False, num_workers=8)

        model.train()
        model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        for epoch in range(self.epochs):
            total_loss = 0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                opt.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                opt.step()

                total_loss += loss.item()

            if epoch % self.verbosity == 0 and epoch != 0:
                print(f'Loss for epoch {epoch} is: {total_loss}')