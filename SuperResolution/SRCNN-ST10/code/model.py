import numpy as np 
import pandas as pd
import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=64,
            kernel_size=9, 
            padding="same"
            )

        self.conv2 = nn.Conv2d(
            in_channels=64, 
            out_channels=32,
            kernel_size=1, 
            padding="same"
            )

        self.conv3 = nn.Conv2d(
            in_channels=32, 
            out_channels=out_channels,
            kernel_size=5, 
            padding="same"
            )

        self.relu = nn.ReLU()

    def forward(self, x) -> None:
        layer_1 = self.conv1(x)
        layer_1 = self.relu(layer_1)
        layer_2 = self.conv2(layer_1)
        layer_2 = self.relu(layer_2)
        layer_3 = self.conv3(layer_2)
        layer_3 = self.relu(layer_3)
        return layer_3 