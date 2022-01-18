import torch
import sys
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=hparams.kernel_size),
            nn.ReLU(),
            nn.Conv2d(6, 24, kernel_size=hparams.kernel_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64,10),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

