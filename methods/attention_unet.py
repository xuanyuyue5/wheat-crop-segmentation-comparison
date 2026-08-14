from __future__ import print_function, division

import torch
import torch.nn as nn

from methods.dl_common import DLSegmentationMethod, WrappedSegmentationModel


class AttentionUNetMethod(DLSegmentationMethod):
    def __init__(self, use_augmentation=True):
        super().__init__(
            model=attention_unet(),
            model_name="attention_unet",
            pretrained=False,
            use_augmentation=use_augmentation,
        )


def attention_unet():
    # Keep the custom attention U-Net lightweight so it fits the shared training pipeline.
    network = AttentionUNet(in_channels=3, num_classes=2)
    return WrappedSegmentationModel("attention_unet", network)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Re-weight skip features before concatenation with the decoder branch.
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class UpAttentionBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, out_channels // 2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attention(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, features=(64, 128, 256, 512)):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_channels, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])

        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        self.dec4 = UpAttentionBlock(features[3] * 2, features[3], features[3])
        self.dec3 = UpAttentionBlock(features[3], features[2], features[2])
        self.dec2 = UpAttentionBlock(features[2], features[1], features[1])
        self.dec1 = UpAttentionBlock(features[1], features[0], features[0])

        self.head = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.dec4(bottleneck, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.head(d1)
