import torch
import torch.nn as nn
import torch.nn.functional as F

# Import modules from modules folder
from modules import SPPInceptionModule, GhostModule, ELAModule, AttentionGate

class MSGUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super(MSGUNet, self).__init__()

        # Encoder channels
        C1, C2, C3, C4, C5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )

        # Encoder path
        self.enc1 = nn.Sequential(
            SPPInceptionModule(in_channels, C1),
            GhostModule(C1, C1),
            ELAModule(C1),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            SPPInceptionModule(C1, C2),
            GhostModule(C2, C2),
            ELAModule(C2),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            SPPInceptionModule(C2, C3),
            GhostModule(C3, C3),
            ELAModule(C3),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            SPPInceptionModule(C3, C4),
            GhostModule(C4, C4),
            ELAModule(C4),
        )
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            SPPInceptionModule(C4, C5),
            GhostModule(C5, C5),
            ELAModule(C5),
        )

        # Decoder path with attention gates
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ag4 = AttentionGate(F_g=C5, F_l=C4, F_int=C4 // 2)
        self.dec4 = GhostModule(C5 + C4, C4)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ag3 = AttentionGate(F_g=C4, F_l=C3, F_int=C3 // 2)
        self.dec3 = GhostModule(C4 + C3, C3)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ag2 = AttentionGate(F_g=C3, F_l=C2, F_int=C2 // 2)
        self.dec2 = GhostModule(C3 + C2, C2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.ag1 = AttentionGate(F_g=C2, F_l=C1, F_int=C1 // 2)
        self.dec1 = GhostModule(C2 + C1, C1)

        # Final output layer
        self.final_conv = nn.Conv2d(C1, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        # Decoder with attention
        d4 = self.up4(b)
        e4_att, _ = self.ag4(e4, d4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att, _ = self.ag3(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att, _ = self.ag2(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att, _ = self.ag1(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        out = self.final_conv(d1)
        return out


if __name__ == "__main__":
    import torch

    # Initialize model
    model = MSGUNet(in_channels=3, out_channels=1, base_channels=32)  # try 16 instead of 32
    x = torch.randn(2, 3, 256, 256)

    print("Input :", x.shape)

    # --- Encoder ---
    e1 = model.enc1(x)
    print("After enc1:", e1.shape)   # H, W same
    p1 = model.pool1(e1)
    print("After pool1:", p1.shape) # H/2, W/2

    e2 = model.enc2(p1)
    print("After enc2:", e2.shape)
    p2 = model.pool2(e2)
    print("After pool2:", p2.shape) # H/4, W/4

    e3 = model.enc3(p2)
    print("After enc3:", e3.shape)
    p3 = model.pool3(e3)
    print("After pool3:", p3.shape) # H/8, W/8

    e4 = model.enc4(p3)
    print("After enc4:", e4.shape)
    p4 = model.pool4(e4)
    print("After pool4:", p4.shape) # H/16, W/16

    b = model.bottleneck(p4)
    print("Bottleneck :", b.shape)  # deepest layer

    # --- Decoder ---
    d4_up = model.up4(b)
    print("Up4 :", d4_up.shape)     # H/8, W/8
    e4_att, _ = model.ag4(e4, d4_up)
    d4 = model.dec4(torch.cat([d4_up, e4_att], dim=1))
    print("After dec4:", d4.shape)

    d3_up = model.up3(d4)
    print("Up3 :", d3_up.shape)     # H/4, W/4
    e3_att, _ = model.ag3(e3, d3_up)
    d3 = model.dec3(torch.cat([d3_up, e3_att], dim=1))
    print("After dec3:", d3.shape)

    d2_up = model.up2(d3)
    print("Up2 :", d2_up.shape)     # H/2, W/2
    e2_att, _ = model.ag2(e2, d2_up)
    d2 = model.dec2(torch.cat([d2_up, e2_att], dim=1))
    print("After dec2:", d2.shape)

    d1_up = model.up1(d2)
    print("Up1 :", d1_up.shape)     # H, W
    e1_att, _ = model.ag1(e1, d1_up)
    d1 = model.dec1(torch.cat([d1_up, e1_att], dim=1))
    print("After dec1:", d1.shape)

    out = model.final_conv(d1)
    print("Output :", out.shape)

