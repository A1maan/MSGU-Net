import torch
import torch.nn as nn


class SPPInceptionModule(nn.Module):
    """
    SPP-Inception Module (paper-faithful implementation).

    Exactly matches Figure 2 from the paper:
    - Branch 1: 1x1 convolution (direct)
    - Branch 2: 3x3 convolution → 3x3 convolution
    - Branch 3: 5x5 convolution → 5x5 convolution
    - Branch 4: 3x3 max pooling → 1x1 convolution

    The first conv in each branch reduces channels from in_channels to out_channels/4,
    then the second conv maintains out_channels/4. This is more parameter-efficient
    than using 1x1 squeeze operations while maintaining the paper's structure.

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

        # Branch 2: 1x1 -> 3x3 (paper faithful: first 3x3 reduces channels)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 -> 5x5 (paper faithful: first 5x5 reduces channels)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Sequential(
                nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
                nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1)
            ),
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
