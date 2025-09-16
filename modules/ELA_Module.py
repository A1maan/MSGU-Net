import torch
import torch.nn as nn
import torch.nn.functional as F


class ELAModule(nn.Module):
    """
    ELA (Enhanced Long-range Attention) Module.
    
    This module is an improvement over the CA (Coordinate Attention) mechanism.
    It replaces 2D Conv and BN layers with 7×1 1D conv and GN (Group Normalization)
    layers to enhance interaction and generalization ability of location information
    embedding.
    
    The ELA mechanism consists of two main steps:
    1. Coordinate information embedding using strip pooling
    2. Coordinate attention generation using 1D convolutions and GN
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for intermediate channels (default: 32)
        kernel_size (int): Kernel size for 1D convolutions (default: 7)
        num_groups (int): Number of groups for Group Normalization (default: 8)
    """
    
    def __init__(self, channels, reduction=32, kernel_size=7, num_groups=8):
        super(ELAModule, self).__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.reduction = reduction
        
        # Calculate intermediate channels with reduction
        self.intermediate_channels = max(channels // reduction, 1)
        
        # Ensure num_groups is compatible with intermediate channels
        self.num_groups = num_groups
        if self.intermediate_channels % self.num_groups != 0:
            self.num_groups = 1  # Fallback to instance normalization
        
        # Horizontal direction processing (for H×1 strip pooling)
        # 1D convolution for height dimension
        self.conv_h = nn.Conv1d(
            in_channels=channels,
            out_channels=self.intermediate_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
        # Group normalization for horizontal direction
        self.gn_h = nn.GroupNorm(
            num_groups=self.num_groups,
            num_channels=self.intermediate_channels
        )
        
        # Vertical direction processing (for 1×W strip pooling)
        # 1D convolution for width dimension
        self.conv_w = nn.Conv1d(
            in_channels=channels,
            out_channels=self.intermediate_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
        # Group normalization for vertical direction
        self.gn_w = nn.GroupNorm(
            num_groups=self.num_groups,
            num_channels=self.intermediate_channels
        )
        
        # Final convolutions to generate attention maps
        self.conv_h_final = nn.Conv1d(
            in_channels=self.intermediate_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False
        )
        
        self.conv_w_final = nn.Conv1d(
            in_channels=self.intermediate_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False
        )
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through ELA module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) with applied attention
        """
        batch_size, channels, height, width = x.size()
        
        # Step 1: Strip pooling - Coordinate information embedding
        
        # Horizontal strip pooling (H, 1) - Equation 3
        # z_h^c(h) = 1/W * Σ(w=0 to W-1) x_c(h, w)
        x_h = torch.mean(x, dim=3, keepdim=False)  # Shape: (B, C, H)
        
        # Vertical strip pooling (1, W) - Equation 4  
        # z_w^c(w) = 1/H * Σ(h=0 to H-1) x_c(h, w)
        x_w = torch.mean(x, dim=2, keepdim=False)  # Shape: (B, C, W)
        
        # Step 2: Coordinate attention generation using 1D convolutions
        
        # Process horizontal features - Equation 5
        # y_h = σ(GN(F_h(z_h)))
        y_h = self.conv_h(x_h)  # Shape: (B, intermediate_channels, H)
        y_h = self.gn_h(y_h)    # Group normalization
        # y_h = F.relu(y_h, inplace=True)  # Nonlinear activation σ
        y_h = self.conv_h_final(y_h)  # Shape: (B, C, H)
        y_h = self.sigmoid(y_h)  # Generate attention weights
        
        # Process vertical features - Equation 6
        # y_w = σ(GN(F_w(z_w)))
        y_w = self.conv_w(x_w)  # Shape: (B, intermediate_channels, W)
        y_w = self.gn_w(y_w)    # Group normalization
        # y_w = F.relu(y_w, inplace=True)  # Nonlinear activation σ
        y_w = self.conv_w_final(y_w)  # Shape: (B, C, W)
        y_w = self.sigmoid(y_w)  # Generate attention weights
        
        # Step 3: Apply coordinate attention - Equation 7
        # Y = x × y_h × y_w
        
        # Reshape attention maps to match input dimensions
        y_h = y_h.unsqueeze(3)  # Shape: (B, C, H, 1)
        y_w = y_w.unsqueeze(2)  # Shape: (B, C, 1, W)
        
        # Apply attention: element-wise multiplication
        output = x * y_h * y_w  # Broadcasting: (B, C, H, W) × (B, C, H, 1) × (B, C, 1, W)
        
        return output
    
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
    ela_module = ELAModule(channels=64, reduction=32, kernel_size=7)
    test_input = torch.randn(2, 64, 32, 32)
    output = ela_module(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {ela_module.get_parameter_count():,}")
    
    print("\nTesting different input sizes...")
    
    # Test with different input sizes
    test_sizes = [(1, 128, 16, 16), (3, 256, 64, 64), (4, 32, 8, 8)]
    
    for batch, channels, height, width in test_sizes:
        ela_test = ELAModule(channels=channels, reduction=16)
        test_input = torch.randn(batch, channels, height, width)
        output = ela_test(test_input)
        
        print(f"Input {test_input.shape} → Output {output.shape}")
        assert test_input.shape == output.shape, "Shape mismatch!"
    
    print("\n✅ All ELA module tests passed!")