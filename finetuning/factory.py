import torch
from torch.utils.data import DataLoader

def fit(model, train_dataloader, val_dataloader, lr, epochs, device, verbosity):
    
    model.train()
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        if epoch % verbosity == 0 and epoch != 0:
            print(f'Loss for epoch {epoch} is: {total_loss}')

def fine_tuning_factory(model, train, device, cfg):
    lr = cfg['lr']
    epochs = cfg['epochs']
    verbosity = cfg['verbosity']
    batch_size = cfg['batch_size']

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8)

    fit(model, train_dataloader, None, lr, epochs, device, verbosity)