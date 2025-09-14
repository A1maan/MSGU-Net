import torch
import torch.nn as nn
import torch.nn.functional as F


class DFCAttention(nn.Module):
    """
    DFC (Decoupled Fully Connected) Attention mechanism.
    
    This is the key component that distinguishes GhostNetV2 from GhostNet.
    It addresses the limitation of local information modeling by introducing
    spatial attention with horizontal and vertical FC layers.
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for channel compression
    """
    
    def __init__(self, channels, reduction=4):
        super(DFCAttention, self).__init__()
        
        self.channels = channels
        reduced_channels = max(channels // reduction, 1)
        
        # Average pooling for downsampling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 1x1 convolution for channel reduction
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # Horizontal FC (1, 5) - models horizontal dependencies
        self.horizontal_fc = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=(1, 5), 
                     padding=(0, 2), groups=reduced_channels, bias=False),
            nn.BatchNorm2d(reduced_channels)
        )
        
        # Vertical FC (5, 1) - models vertical dependencies
        self.vertical_fc = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=(5, 1), 
                     padding=(2, 0), groups=reduced_channels, bias=False),
            nn.BatchNorm2d(reduced_channels)
        )
        
        # Final convolution to restore channel dimension
        self.conv_expand = nn.Sequential(
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # Sigmoid activation for attention weights
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through DFC attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Attention weights of shape (B, C, H, W)
        """
        batch_size, channels, height, width = x.size()
        
        # Step 1: Average pooling for downsampling
        pooled = self.avg_pool(x)  # Shape: (B, C, 1, 1)
        
        # Step 2: Channel reduction with 1x1 convolution
        reduced = self.conv_reduce(pooled)  # Shape: (B, C//reduction, 1, 1)
        
        # # Step 3: Broadcast to full spatial size
        # reduced_map = reduced.expand(-1, -1, height, width)  # Shape: (B, C//reduction, H, W)
        
        # Step 4: Apply horizontal and vertical convolutions
        horizontal_out = self.horizontal_fc(reduced)  # Shape: (B, C//reduction, H, W)
        vertical_out = self.vertical_fc(reduced)      # Shape: (B, C//reduction, H, W)
        
        # Step 5: Combine horizontal and vertical features
        combined = horizontal_out + vertical_out  # Shape: (B, C//reduction, H, W)
        
        # Step 6: Restore channel dimension
        attention = self.conv_expand(combined)  # Shape: (B, C, H, W)
        
        # Step 7: Apply sigmoid activation
        attention = self.sigmoid(attention)
        
        # Step 8: Upsample back to original input size (paper: bilinear interpolation)
        attention = F.interpolate(attention, size=(height, width), mode='bilinear', align_corners=False)
        
        return attention


class GhostModule(nn.Module):
    """
    Ghost Module from GhostNetV2.
    
    This module generates more feature maps using fewer parameters by:
    1. Using 1x1 pointwise convolution for primary features
    2. Using 3x3 depthwise convolution for ghost features
    3. Applying DFC attention mechanism for spatial dependencies
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size for depthwise convolution (default: 3)
        ratio (int): Ratio for splitting primary and ghost features (default: 2)
        dw_size (int): Kernel size for depthwise convolution (default: 3)
        stride (int): Stride for the convolution (default: 1)
        use_attention (bool): Whether to use DFC attention (default: True)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, 
                 dw_size=3, stride=1, use_attention=True):
        super(GhostModule, self).__init__()
        
        self.out_channels = out_channels
        self.ratio = ratio
        self.use_attention = use_attention
        
        # Calculate primary feature channels
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)
        
        # Part 1: 1x1 pointwise convolution (Primary features)
        # Y' = F1x1 * X (Equation 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, 
                     kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        # Part 2: 3x3 depthwise convolution (Ghost features)
        # Y'' = Concat(Y', Fdp * Y') (Equation 2)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, 
                     groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )
        
        # Part 3: DFC Attention mechanism (GhostNetV2 improvement)
        if self.use_attention:
            self.dfc_attention = DFCAttention(out_channels)
    
    def forward(self, x):
        """
        Forward pass through Ghost module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W)
        """
        # Part 1: Generate primary features using 1x1 pointwise convolution
        primary_features = self.primary_conv(x)  # Y' in Equation 1
        
        # Part 2: Generate ghost features using 3x3 depthwise convolution
        ghost_features = self.cheap_operation(primary_features)  # Fdp * Y'
        
        # Concatenate primary and ghost features (Equation 2)
        output = torch.cat([primary_features, ghost_features], dim=1)  # Y''
        
        # Part 3: Apply DFC attention mechanism (GhostNetV2 improvement)
        if self.use_attention:
            attention_weights = self.dfc_attention(output)
            output = output * attention_weights  # Element-wise multiplication
        
        return output
    
    def get_parameter_count(self):
        """
        Calculate the total number of parameters in this module.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test the Ghost Module implementation
    print("Testing Ghost Module...")
    
    # Test basic Ghost Module
    ghost_module = GhostModule(in_channels=64, out_channels=128, use_attention=True)
    test_input = torch.randn(2, 64, 32, 32)
    output = ghost_module(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {ghost_module.get_parameter_count():,}")
    
    print("\nTesting Ghost Module without attention...")
    
    # Test Ghost Module without attention (original GhostNet)
    ghost_module_no_att = GhostModule(in_channels=64, out_channels=128, use_attention=False)
    output_no_att = ghost_module_no_att(test_input)
    
    print(f"Output shape (no attention): {output_no_att.shape}")
    print(f"Parameters (no attention): {ghost_module_no_att.get_parameter_count():,}")
    
    print("\nTesting DFC Attention separately...")
    
    # Test DFC Attention mechanism
    dfc_attention = DFCAttention(channels=128)
    attention_weights = dfc_attention(output)
    
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention range: [{attention_weights.min().item():.4f}, {attention_weights.max().item():.4f}]")