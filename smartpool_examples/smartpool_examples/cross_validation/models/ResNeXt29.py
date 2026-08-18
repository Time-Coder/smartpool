import torch.nn as nn


class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality=32, base_width=4, stride=1):
        super().__init__()
        self.width = int(out_channels * (base_width / 64)) * cardinality

        self.conv1 = nn.Conv2d(in_channels, self.width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width)
        self.conv2 = nn.Conv2d(self.width, self.width, 3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(self.width)
        self.conv3 = nn.Conv2d(self.width, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class ResNeXt29(nn.Module):
    def __init__(self, cardinality=32, base_width=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 256, 3, stride=1, cardinality=cardinality, base_width=base_width)
        self.layer2 = self._make_layer(256, 512, 3, stride=2, cardinality=cardinality, base_width=base_width)
        self.layer3 = self._make_layer(512, 1024, 3, stride=2, cardinality=cardinality, base_width=base_width)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 10)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, cardinality=32, base_width=4):
        layers = []
        layers.append(ResNeXtBlock(in_channels, out_channels, cardinality, base_width, stride))
        for _ in range(1, blocks):
            layers.append(ResNeXtBlock(out_channels, out_channels, cardinality, base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x