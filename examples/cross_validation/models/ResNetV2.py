import torch.nn as nn

class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(identity)
        return out

class ResNetV2(nn.Module):
    def __init__(self):
        super(ResNetV2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        
        self.layer1 = self._make_layer(32, 64, 2, stride=2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(PreActBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(PreActBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x