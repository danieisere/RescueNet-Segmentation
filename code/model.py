import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ========== 通用模块（轻量 conv_block，SE 注意力） ==========
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


# 轻量注意力机制（SE Block）
class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ========== 1. UNet with ResNet18 encoder + Custom Pyramid Module ==========
class CustomPyramidModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), out_channels=512):
        super(CustomPyramidModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        pyramid_feats = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        out = torch.cat([x] + pyramid_feats, dim=1)
        return self.project(out)


class UNet(nn.Module):
    def __init__(self, num_classes=11):
        super(UNet, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # (B, 64, H/2, W/2)
        self.pool0 = resnet.maxpool  # (B, 64, H/4, W/4)
        self.encoder1 = resnet.layer1  # (B, 64, H/4, W/4)
        self.encoder2 = resnet.layer2  # (B, 128, H/8, W/8)
        self.encoder3 = resnet.layer3  # (B, 256, H/16, W/16)
        self.encoder4 = resnet.layer4  # (B, 512, H/32, W/32)

        self.pyramid = CustomPyramidModule(512, out_channels=512)

        self.up4 = up_conv(512, 256)
        self.dec4 = conv_block(512, 256)
        self.se4 = SEBlock(256)

        self.up3 = up_conv(256, 128)
        self.dec3 = conv_block(256, 128)
        self.se3 = SEBlock(128)

        self.up2 = up_conv(128, 64)
        self.dec2 = conv_block(128, 64)
        self.se2 = SEBlock(64)

        self.up1 = up_conv(64, 32)
        self.dec1 = conv_block(96, 32)  # 64 from encoder0 + 32 upsampled
        self.se1 = SEBlock(32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool0(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.pyramid(e4)

        d4 = self.up4(center)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        d4 = self.se4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d3 = self.se3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        d2 = self.se2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.dec1(d1)
        d1 = self.se1(d1)

        out = self.final(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)  # 新增
        return out


# ========== 2. PSPNet ==========
class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), reduction_channels=512):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=ps),
                nn.Conv2d(in_channels, reduction_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_channels),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * reduction_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x] + [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes=11, backbone='resnet50'):
        super(PSPNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.ppm = PSPModule(in_channels=512)
        self.final = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ppm(x)
        x = self.final(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x


# ========== 3. DeepLabV3 ==========
# ----------------模型3：DeepLabV3 ---------------- #
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for rate in atrous_rates:
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = [F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=True)]
        for stage in self.stages:
            feats.append(stage(x))
        x = torch.cat(feats, dim=1)
        return self.project(x)


class DeepLabV3_Custom(nn.Module):
    def __init__(self, num_classes=11, backbone='resnet50', pretrained=True):
        super(DeepLabV3_Custom, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = ASPP(in_channels=512)
        self.se = SEBlock(256)  # 引入注意力机制

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        size = x.size()[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = self.se(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x


# ----------------统一加载接口---------------- #
def get_model(model_name, num_classes=11):
    if model_name == 'unet':
        return UNet(num_classes=num_classes)
    elif model_name == 'pspnet':
        return PSPNet(num_classes=num_classes, backbone='resnet18')
    elif model_name == 'deeplabv3':
        return DeepLabV3_Custom(num_classes=num_classes, backbone='resnet18')
    else:
        raise ValueError(f"Unknown model name {model_name}, expected 'unet', 'pspnet' or 'deeplabv3'.")
