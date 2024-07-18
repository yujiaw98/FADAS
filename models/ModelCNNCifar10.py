import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn


class ModelCNNCifar10(nn.Module):
    def __init__(self):
        super(ModelCNNCifar10, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(64, 10)

        # Use Kaiming initialization for layers with ReLU activation
        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        self.conv.apply(init_weights)
        self.fc1.apply(init_weights)

    def forward(self, x):
        conv_ = self.conv(x)
        fc_ = conv_.view(-1, 8 * 8 * 32)
        fc1_ = self.fc1(fc_)
        output = self.fc2(fc1_)
        return output

