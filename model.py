"""U-Net architecture for semantic segmentation."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive (Conv2d → BatchNorm → ReLU) blocks."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Classic U-Net with 4 encoder stages, a bridge, and 4 decoder stages.

    Architecture:
        Input (3ch) → Encoder [64→128→256→512]
                     → Bridge [1024]
                     → Decoder [512→256→128→64] (with skip connections)
                     → Output (num_classes)
    """

    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()

        # Encoder
        self.d1 = DoubleConv(in_channels, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bridge
        self.bridge = DoubleConv(512, 1024)

        # Decoder
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.c4 = DoubleConv(1024, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c3 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c1 = DoubleConv(128, 64)

        # Output
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder path
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        # Bridge
        b = self.bridge(self.pool(d4))

        # Decoder path with skip connections
        u4 = self.c4(torch.cat([self.u4(b), d4], dim=1))
        u3 = self.c3(torch.cat([self.u3(u4), d3], dim=1))
        u2 = self.c2(torch.cat([self.u2(u3), d2], dim=1))
        u1 = self.c1(torch.cat([self.u1(u2), d1], dim=1))

        return self.out(u1)
