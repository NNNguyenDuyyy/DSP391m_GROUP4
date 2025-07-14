import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.dw_conv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.pw_conv(x)
        return x * u

class VGG16_LKA(nn.Module):
    def __init__(self, num_classes, dropout=None):
        super().__init__()
        base = models.vgg16_bn(pretrained=True)
        self.block1 = nn.Sequential(*base.features[0:6])    # 64
        self.block2 = nn.Sequential(*base.features[7:13])   # 128
        self.block3 = nn.Sequential(*base.features[14:23])  # 256
        self.block4 = nn.Sequential(*base.features[24:33])  # 512
        self.block5 = nn.Sequential(*base.features[34:43])  # 512
        self.lka3 = LKA(dim=256)
        self.lka4 = LKA(dim=512)
        self.lka5 = LKA(dim=512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.block3(x)
        x = self.lka3(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.block4(x)
        x = self.lka4(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.block5(x)
        x = self.lka5(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x 