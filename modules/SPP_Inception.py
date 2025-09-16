import torch
import torch.nn as nn


class SPPInceptionModule(nn.Module):
    """
    SPP-Inception Module (as described in the paper).

    - Branch 1: 1x1 convolution
    - Branch 2: 3x3 convolution followed by 3x3 convolution
    - Branch 3: 5x5 convolution followed by 5x5 convolution
    - Branch 4: 3x3 max pooling followed by 1x1 convolution

    Each branch outputs out_channels / 4, so after concatenation
    the total equals out_channels.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (must be divisible by 4)
    """

    def __init__(self, in_channels, out_channels):
        super(SPPInceptionModule, self).__init__()

        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        branch_channels = out_channels // 4

        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 3x3 maxpool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick test
    model = SPPInceptionModule(in_channels=64, out_channels=128)
    x = torch.randn(2, 64, 32, 32)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters:   {model.get_parameter_count():,}")
