import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torch.nn.quantized import FloatFunctional


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.residual_add = FloatFunctional()

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y = self.residual_add.add(Y, X)
        return self.relu(Y)
    
    def fuse_modules(self):
        quant.fuse_modules(self, ['conv1', 'bn1'], inplace=True)
        quant.fuse_modules(self, ['conv2', 'bn2'], inplace=True)
    
    def fuse_modules_qat(self):
        quant.fuse_modules_qat(self, ['conv1', 'bn1'], inplace=True)
        quant.fuse_modules_qat(self, ['conv2', 'bn2'], inplace=True)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet_block1 = self.resnet_block(64, 64, num_residuals=2, first_block=True)
        self.resnet_block2 = self.resnet_block(64, 128, num_residuals=2)
        self.resnet_block3 = self.resnet_block(128, 256, num_residuals=2)
        self.resnet_block4 = self.resnet_block(256, 512, num_residuals=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    def resnet_block(self, in_channels, out_channels, num_residuals, first_block=False):
        if first_block:
            assert in_channels == out_channels # 第一个模块的通道数与输入通道数一致
        blocks = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blocks.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blocks.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blocks)

    def forward(self, X):
        X = self.quant(X)
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)

        X = self.resnet_block1(X)
        X = self.resnet_block2(X)
        X = self.resnet_block3(X)
        X = self.resnet_block4(X)

        X = self.avg_pool(X)
        X = torch.flatten(X, 1)
        X = self.fc(X)
        X = self.dequant(X)
        return X
    
    def fuse_model(self):
        quant.fuse_modules(self, ['conv1', 'bn1'], inplace=True)
        for i in range(len(self.resnet_block1)):
            self.resnet_block1[i].fuse_modules()
        for i in range(len(self.resnet_block2)):
            self.resnet_block2[i].fuse_modules()
        for i in range(len(self.resnet_block3)):
            self.resnet_block3[i].fuse_modules()
        for i in range(len(self.resnet_block4)):
            self.resnet_block4[i].fuse_modules()
    
    def fuse_model_qat(self):
        quant.fuse_modules_qat(self, ['conv1', 'bn1'], inplace=True)
        for i in range(len(self.resnet_block1)):
            self.resnet_block1[i].fuse_modules_qat()
        for i in range(len(self.resnet_block2)):
            self.resnet_block2[i].fuse_modules_qat()
        for i in range(len(self.resnet_block3)):
            self.resnet_block3[i].fuse_modules_qat()
        for i in range(len(self.resnet_block4)):
            self.resnet_block4[i].fuse_modules_qat()