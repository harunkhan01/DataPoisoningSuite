import torch

class FineTuningClass:
    def __init__(self, dl_obj, model_obj, **kwargs):
        self.epochs = kwargs.get('epochs', 10)
        self.lr = kwargs.get('lr', 1e-5)
        self.device = kwargs.get('device', 'cuda')
        self.dl_obj = dl_obj
        self.model_obj = model_obj

    def fine_tune(self):
        train_dl = self.dl_obj.train_dl

        model = self.model_obj.model
        model.train()
        model.to(self.device)

        loss = torch.nn.modules.loss.CrossEntropyLoss()
        opt = torch.optim.sgd.SGD(model.parameters(), lr=self.lr)

        for epoch in self.epochs:
            total_loss = 0

            for x, y in train_dl:
                x.to(self.device)
                y.to(self.device)

                opt.zero_grad()
                output = model(x)
                total_loss += loss(output, y)
                loss.backward()
                opt.step()
            
            if epoch % 10 == 0 and epoch > 0:
                print(f'Total loss for epoch {epoch} is : {total_loss}')