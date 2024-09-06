import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    network = Network()
    input = torch.ones((64, 3, 32, 32))
    output = network(input)
    print(output.shape)  # Output shape: (64, 10)
