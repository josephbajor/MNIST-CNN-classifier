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
        total_loss = 0.0
        for label, dat in trainloader:

            opt.zero_grad()

            out = model(dat)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Loss for Epoch {epoch+1}: {total_loss}")