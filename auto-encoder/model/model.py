import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


class MnistModel(BaseModel):
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


class AutoEncoder(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # B, 16, 10, 10
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # B, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # B, 8, 3, 3
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)  # B, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # B, 16, 5, 5
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # B， 8， 15， 15
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # B, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode