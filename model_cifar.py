import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output




class CIFARNet(nn.Module):
    """
    C1–C4–Output backbone for CIFAR-10 with residual connections and spatial dropout.
    
    Channel progression: 3 -> 40 -> 128 -> 240 -> 384 -> 10
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.1, drop_path_rate: float = 0.1):
        super().__init__()

        # C1: Large kernel to boost receptive field early (3->40)
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p*0.5),
        )

        # C2: Depthwise separable convolution with residual (40->128)
        self.c2_depthwise = nn.Conv2d(40, 40, kernel_size=3, padding=1, groups=40, bias=False)
        self.c2_pointwise = nn.Conv2d(40, 128, kernel_size=1, bias=False)
        self.c2_bn = nn.BatchNorm2d(128)
        self.c2_drop = nn.Dropout2d(dropout_p)
        
        # Residual connection for C2 (40->128)
        self.c2_residual = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128)
        )

        # C3: Dilated depthwise separable conv with residual (128->240)
        # For 3x3 kernel with dilation=4: effective kernel = 9x9, padding = 4
        self.c3_depthwise = nn.Conv2d(128, 128, kernel_size=3, dilation=4, padding=4, groups=128, bias=False)
        self.c3_pointwise = nn.Conv2d(128, 240, kernel_size=1, bias=False)
        self.c3_bn = nn.BatchNorm2d(240)
        self.c3_drop = nn.Dropout2d(dropout_p)
        
        # Residual connection for C3 (128->240)
        self.c3_residual = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=240, kernel_size=1, bias=False),
            nn.BatchNorm2d(240)
        )

        # C4: Stride-2 depthwise separable conv with residual (240->384)
        self.c4_depthwise = nn.Conv2d(240, 240, kernel_size=3, stride=2, padding=1, groups=240, bias=False)
        self.c4_pointwise = nn.Conv2d(240, 384, kernel_size=1, bias=False)
        self.c4_bn = nn.BatchNorm2d(384)
        self.c4_drop = nn.Dropout2d(dropout_p*1.2)
        
        # Output: Fully connected layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(384, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # C1
        x = self.c1(x)
        
        # C2 depthwise separable with residual
        identity_c2 = self.c2_residual(x)
        x = self.c2_depthwise(x)
        x = self.c2_pointwise(x)
        x = self.c2_bn(x)
        x = F.relu(x + identity_c2, inplace=True)
        x = self.c2_drop(x)

        # C3 dilated depthwise separable with residual
        identity_c3 = self.c3_residual(x)
        x = self.c3_depthwise(x)
        x = self.c3_pointwise(x)
        x = self.c3_bn(x)
        x = F.relu(x + identity_c3, inplace=True)
        x = self.c3_drop(x)
        
        # C4 stride-2 depthwise separable with residual
        x = self.c4_depthwise(x)
        x = self.c4_pointwise(x)
        x = self.c4_bn(x)
        x = F.relu(x, inplace=True)

        # Output: Global average pooling + FC layer
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def build_model(device: torch.device, num_classes: int = 10) -> nn.Module:
    return CIFARNet(num_classes=num_classes).to(device)



