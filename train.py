import torch
import torch.nn as nn
from model import ConvModel
from dataloaders import load_data

def train(hparams):
    model = ConvModel(hparams).cuda()
    
    opt = torch.optim.Adam(model.parameters(), lr=hparams.learn_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    trainloader, testloader = load_data(hparams)


    for epoch in range(hparams.epochs):
        for label, dat in trainloader:

            opt.zero_grad()

            out = model(dat)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()

            print(f"batch loss: {loss.item()}")

    print(f"Epoch: {epoch}")