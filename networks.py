from torch import nn
import torch.nn.functional as F


class fl_net(nn.Module):
    def __init__(self, in_chan, in_d, out_d):
        super().__init__()
        d = int((in_d - 3 * 2 + 2) / 2)  # output size after 2 convs and maxpools with kernel 3
        self.conv1 = nn.Conv2d(in_chan, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.fc1 = nn.Linear(64*d*d, 128)
        self.fc2 = nn.Linear(128, out_d)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=(2, 2))(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = nn.Dropout(p=0.5)(x)
        x = F.relu(self.fc1(x))
        x = nn.Dropout(p=0.5)(x)
        x = self.fc2(x)
        return x
