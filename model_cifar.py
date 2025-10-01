import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARNet(nn.Module):
    """
    C1–C4–Output backbone for CIFAR-10.

    Receptive field (RF) calculation (stride listed where applicable):
      - C1: 11x11 conv → RF = 1 + (11-1)*1 = 11, jump j = 1
      - C2: 7x7 depthwise (s=1) → RF = 11 + (7-1)*1 = 17
           (1x1 pointwise does not change RF)
      - C3: 7x7 depthwise with dilation=4 → effective k = 7 + (7-1)*(4-1) = 25
           RF = 17 + (25-1)*1 = 41 (1x1 pointwise no change)
      - C4: 5x5 depthwise with stride=2 → RF = 41 + (5-1)*1 = 45 (>44), new j = 2
      - Classifier 1x1 + GAP do not change RF
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.05):
        super().__init__()

        # C1: Large kernel to boost receptive field early (3->32)
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

        # C2: Depthwise separable convolution (32->128)
        self.c2_depthwise = nn.Conv2d(32, 32, kernel_size=7, padding=3, groups=32, bias=False)
        self.c2_pointwise = nn.Conv2d(32, 128, kernel_size=1, bias=False)
        self.c2_bn = nn.BatchNorm2d(128)
        self.c2_drop = nn.Dropout(dropout_p)

        # C3: Dilated depthwise separable conv (128->256)
        # kernel=7, dilation=4 -> effective kernel=25, padding=12 keeps size
        self.c3_depthwise = nn.Conv2d(128, 128, kernel_size=7, dilation=4, padding=12, groups=128, bias=False)
        self.c3_pointwise = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.c3_bn = nn.BatchNorm2d(256)
        self.c3_drop = nn.Dropout(dropout_p)

        # C4: Stride-2 depthwise separable conv (256->448)
        self.c4_depthwise = nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2, groups=256, bias=False)
        self.c4_pointwise = nn.Conv2d(256, 448, kernel_size=1, bias=False)
        self.c4_bn = nn.BatchNorm2d(448)
        self.c4_drop = nn.Dropout(dropout_p)

        # Output: 1x1 conv to map to classes, followed by GAP
        self.classifier = nn.Conv2d(448, num_classes, kernel_size=1, bias=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # C1
        x = self.c1(x)

        # C2 depthwise separable
        x = F.relu(self.c2_bn(self.c2_pointwise(self.c2_depthwise(x))), inplace=True)
        x = self.c2_drop(x)

        # C3 dilated depthwise separable
        x = F.relu(self.c3_bn(self.c3_pointwise(self.c3_depthwise(x))), inplace=True)
        x = self.c3_drop(x)

        # C4 stride-2 depthwise separable
        x = F.relu(self.c4_bn(self.c4_pointwise(self.c4_depthwise(x))), inplace=True)
        x = self.c4_drop(x)

        # Output
        x = self.classifier(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


def build_model(device: torch.device, num_classes: int = 10) -> nn.Module:
    return CIFARNet(num_classes=num_classes).to(device)


