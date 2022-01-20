import os
import torch
import torch.nn as nn
from model import ConvModel
from dataloaders import load_data

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def train(hparams, load:bool = False):
    savepath = os.path.join(__location__, hparams.model_path)
    model = ConvModel(hparams).cuda()

    if load:
        assert os.path.isfile(savepath), f"Cannot find {savepath}!"
        model.load_state_dict(torch.load(savepath))
    
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

        print(f"\nLoss for Epoch {epoch+1}: {total_loss}")
        testiter = iter(testloader)
        label, dat = testiter.next()
        _, out = torch.max(model(dat), 1)
        acc = (label == out).sum().item()/hparams.batch_size * 100
        print(f"{acc}% Accuracy")

    torch.save(model.state_dict(), savepath)
    print(f"Saving model parameters to {savepath}")