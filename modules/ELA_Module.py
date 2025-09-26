import torch
import torch.nn as nn
import torch.nn.functional as F


class ELAModule(nn.Module):
    """
    ELA (Enhanced Long-range Attention) Module - Efficient Implementation.
    
    This module implements the efficient localization attention mechanism.
    It uses 1D convolutions with Group Normalization for coordinate attention
    generation without intermediate channel reduction for better efficiency.
    
    The ELA mechanism consists of:
    1. Strip pooling for coordinate information embedding
    2. 1D convolutions with Group Normalization for attention generation
    3. Element-wise multiplication for attention application
    
    Args:
        channel (int): Number of input channels
        kernel_size (int): Kernel size for 1D convolutions (default: 7)
    """
    
    def __init__(self, channel, kernel_size=7):
        super(ELAModule, self).__init__()
        
        self.channel = channel
        self.kernel_size = kernel_size
        
        # Calculate padding for 1D convolutions
        pad = kernel_size // 2
        
        # 1D convolution for horizontal direction (height pooling)
        self.conv_h = nn.Conv1d(
            channel, 
            channel, 
            kernel_size=kernel_size,
            padding=pad, 
            groups=channel, 
            bias=False
        )
        
        # Group normalization for horizontal direction
        self.gn_h = nn.GroupNorm(16, channel)
        
        # 1D convolution for vertical direction (width pooling)
        self.conv_w = nn.Conv1d(
            channel, 
            channel, 
            kernel_size=kernel_size,
            padding=pad, 
            groups=channel, 
            bias=False
        )
        
        # Group normalization for vertical direction
        self.gn_w = nn.GroupNorm(16, channel)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through ELA module - Efficient Implementation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) with applied attention
        """
        b, c, h, w = x.size()
        
        # Strip pooling for coordinate information embedding
        # Horizontal pooling: average across width dimension
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)  # Shape: (B, C, H)
        
        # Vertical pooling: average across height dimension  
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)  # Shape: (B, C, W)
        
        # Generate attention weights for horizontal direction
        x_h = self.sigmoid(self.gn_h(self.conv_h(x_h))).view(b, c, h, 1)
        
        # Generate attention weights for vertical direction
        x_w = self.sigmoid(self.gn_w(self.conv_w(x_w))).view(b, c, 1, w)
        
        # Apply coordinate attention through element-wise multiplication
        return x * x_h * x_w
    
    def get_parameter_count(self):
        """
        Calculate the total number of parameters in this module.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    # Test the ELA Module implementation
    print("Testing ELA Module...")
    
    # Test basic ELA Module
    ela_module = ELAModule(channel=64, kernel_size=7)
    test_input = torch.randn(2, 64, 32, 32)
    output = ela_module(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {ela_module.get_parameter_count():,}")
    
    print("\nTesting different input sizes...")
    
    # Test with different input sizes
    test_sizes = [(1, 128, 16, 16), (3, 256, 64, 64), (4, 32, 8, 8)]
    
    for batch, channels, height, width in test_sizes:
        ela_test = ELAModule(channel=channels, kernel_size=7)
        test_input = torch.randn(batch, channels, height, width)
        output = ela_test(test_input)
        
        print(f"Input {test_input.shape} → Output {output.shape}")
        assert test_input.shape == output.shape, "Shape mismatch!"
    
    print("\n✅ All ELA module tests passed!")