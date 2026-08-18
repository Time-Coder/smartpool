import torch.nn as nn


class WideBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
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


class WideResNet2810(nn.Module):
    def __init__(self):
        super().__init__()
        base_width = 16
        widen_factor = 10

        w0 = base_width
        w1 = base_width * widen_factor
        w2 = base_width * 2 * widen_factor
        w3 = base_width * 4 * widen_factor

        self.conv1 = nn.Conv2d(3, w0, 3, padding=1, bias=False)

        self.layer1 = self._make_layer(w0, w1, 4, stride=1)
        self.layer2 = self._make_layer(w1, w2, 4, stride=2)
        self.layer3 = self._make_layer(w2, w3, 4, stride=2)

        self.bn = nn.BatchNorm2d(w3)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w3, 10)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(WideBasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(WideBasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x