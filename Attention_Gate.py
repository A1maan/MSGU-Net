import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate from MSGU-Net (based on Attention U-Net).
    
    Args:
        F_g (int): Channels of gating signal (decoder input)
        F_l (int): Channels of skip connection (encoder input)
        F_int (int): Intermediate channel size after 1x1 conv
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        # 1x1 conv for gating signal (decoder features)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 1x1 conv for encoder skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Combine + ReLU
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x, g):
        # x: encoder features (skip connection)
        # g: gating features (decoder)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # attention map (B,1,H,W)
        
        # Apply attention coefficients
        out = x * psi
        return out, psi
    
if __name__ == "__main__":
    import torch

    # Example sizes (like in U-Net):
    B = 2            # batch size
    F_g = 256        # decoder gating channels
    F_l = 128        # encoder skip connection channels
    F_int = 64       # intermediate channels
    H, W = 32, 32    # spatial size

    # Initialize Attention Gate
    ag = AttentionGate(F_g=F_g, F_l=F_l, F_int=F_int)

    # Fake input tensors
    x = torch.randn(B, F_l, H, W)        # encoder skip features
    g = torch.randn(B, F_g, H, W)        # decoder gating features (upsampled to same H,W)

    # Forward pass
    out, att_map = ag(x, g)

    print("Input skip x:", x.shape)
    print("Input gate g:", g.shape)
    print("Attention map:", att_map.shape)
    print("Output gated features:", out.shape)

    # Sanity checks
    assert out.shape == x.shape, "❌ Output shape mismatch!"
    assert att_map.shape == (B, 1, H, W), "❌ Attention map shape mismatch!"

    print("✅ AttentionGate test passed!")

