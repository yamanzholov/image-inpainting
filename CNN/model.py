import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=5)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        return x

