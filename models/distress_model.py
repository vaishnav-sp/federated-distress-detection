import torch
import torch.nn as nn


class DistressModel(nn.Module):

    def __init__(self, input_size=24, num_classes=2):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.network(x)