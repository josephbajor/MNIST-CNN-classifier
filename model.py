import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

class ConvModelold(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=hparams.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 24, kernel_size=hparams.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(1),
            nn.Linear(24*5*5,48),
            nn.ReLU(),
            nn.Linear(48,10),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


class ConvModel(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, hparams.kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, hparams.kernel_size)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
