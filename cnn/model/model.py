import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv3x3(1, 16)
        self.conv2 = Conv3x3(16, 32)
        self.conv3 = Conv3x3(32, 64)
        self.pool = nn.MaxPool2d(3, 2, 1)
        # self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.gap = nn.Conv2d(64, 64, 7, 7, groups=64, bias=False)
        self.fc1 = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(-1, 64)
        
        x = self.drop(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class MnistMLP(BaseModel):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)