import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  #28x28x32
        self.pool = nn.MaxPool2d(2, 2)  #14x14x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  #14x14x64 converts to 7x7x64- don't have to define the pool layer again
        self.fc1 = nn.Linear(7*7*64, 128)  
        self.fc2 = nn.Linear(128, 10)  
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  
        return x
