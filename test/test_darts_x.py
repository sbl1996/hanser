import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.alphas = torch.randn(3, 4, requires_grad=True) * 1e-3

    def forward(self, x):
        return self.fc1(x) @ self.alphas

net = Net()
x = torch.randn(4, 2)
net(x)