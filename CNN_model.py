import itertools
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image  # 用于图像显示
from Denoising import medianblur  # 调用去噪库中的中值滤波函数


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 输入通道=3，输出通道=32 ,卷积核大小5*5，步长为1，填充为2
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),  # 添加 BatchNorm
            # nn.Sigmoid(),
            nn.LeakyReLU(negative_slope=0.01),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：128*128 到 64*64

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32到64
            nn.BatchNorm2d(64),  # 添加 BatchNorm
            nn.LeakyReLU(negative_slope=0.01),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：64*64 到 32*32

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 64到128
            nn.BatchNorm2d(128),  # 添加 BatchNorm
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：32*32 到 16*16

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 128到256
            nn.BatchNorm2d(64),  # 添加 BatchNorm
            nn.LeakyReLU(negative_slope=0.01),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：16x16 到 8x8
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 展平特征图
            nn.Linear(64*8*8, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),  # Dropout 防止过拟合
            nn.Linear(64, num_classes),  # 输出层，分类数=num_classes
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 通过卷积层
        x = self.fc_layers(x)  # 通过全连接层
        return x
