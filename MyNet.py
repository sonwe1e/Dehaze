import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.down = nn.PixelUnshuffle(4)
        self.conv1 = Conv(in_channel * 16, out_channel)
        self.conv2 = Conv(out_channel, out_channel * 16)
        self.up = nn.PixelShuffle(4)

    def forward(self, x):
        x = self.down(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up(x)
        return x

class A(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv1 = Conv(in_channel, out_channel)
        self.conv2 = Conv(out_channel, out_channel)
        self.conv3 = Conv(out_channel, out_channel)
        self.conv4 = Conv(out_channel, out_channel)
        self.pool = nn.MaxPool2d(2, 2)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channel, int(out_channel ** 0.5)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(int(out_channel ** 0.5), 1)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.pool(x4)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = Conv(3, 16)
        self.conv1 = Conv(16, 16)
        self.conv2 = Conv(16, 32)
        self.conv3 = Conv(32, 64)
        self.out = Conv(64, 3)
        self.down1 = Down(16, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.A1 = A(16, 16)
        self.A2 = A(16, 32)
        self.A3 = A(32, 64)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.conv1(x)
        d1 = self.down1(x)
        A1 = self.A1(x)
        x = d1 * x1 + A1 * (1 - d1)
        x2 = self.conv2(x)
        d2 = self.down2(x)
        A2 = self.A2(x)
        x = d2 * x2 + A2 * (1 - d2)
        x3 = self.conv3(x)
        d3 = self.down3(x)
        A3 = self.A3(x)
        x = d3 * x3 + A3 * (1 - d3)
        x = self.out(x)
        return x


if __name__ == '__main__':
    model = Net()
    print(model)

    x = torch.randn(1, 3, 400, 400)
    y = model(x)
    print(y.shape)