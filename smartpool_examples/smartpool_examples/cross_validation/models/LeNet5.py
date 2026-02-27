import torch.nn as nn


class LeNet5(nn.Module):
    
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),      # 28x28 -> 24x24
            nn.ReLU(),
            nn.MaxPool2d(2),          # 24x24 -> 12x12
            nn.Conv2d(6, 16, 5),      # 12x12 -> 8x8
            nn.ReLU(),
            nn.MaxPool2d(2)           # 8x8 -> 4x4
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)