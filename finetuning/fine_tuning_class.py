import torch

class FineTuningClass:
    def __init__(self, cfg):
        self.epochs = cfg.epochs
        self.lr = cfg.lr
        self.device = cfg.device
        self.verbosity = cfg.verbosity

    def fine_tune(self, model, dataloader):
        model = self.model_obj.model
        model.train()
        model.to(self.device)

        loss = torch.nn.modules.loss.CrossEntropyLoss()
        opt = torch.optim.sgd.SGD(model.parameters(), lr=self.lr)

        for epoch in self.epochs:
            total_loss = 0

            for x, y in dataloader:
                x.to(self.device)
                y.to(self.device)

                opt.zero_grad()
                output = model(x)
                total_loss += loss(output, y)
                loss.backward()
                opt.step()
            
            if epoch % self.verbosity == 0 and epoch > 0:
                print(f'Total loss for epoch {epoch} is : {total_loss}')
    
    def __call__(self, model, dataloader):
        self.fine_tune(model, dataloader)