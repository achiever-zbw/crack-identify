import itertools
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image  # 用于图像显示
from Denoising import medianblur  # 调用去噪库中的中值滤波函数

# 定义IC层


class IC(nn.Module):
    def __init__(self, channels, dropout):
        super(IC, self).__init__()
        self.batchnorm = nn.BatchNorm2d(channels)  # 定义 BatchNorm2d 层
        self.dropout = nn.Dropout2d(dropout)      # 定义 Dropout2d 层

    def forward(self, x):
        x = self.batchnorm(x)  # 批归一化
        x = self.dropout(x)    # Dropout
        return x


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 输入通道=3，输出通道=32 ,卷积核大小5*5，步长为1，填充为2
            IC(3, dropout=0.5),  # 引入IC层置于权重层前
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：128*128 到 64*64

            IC(32, dropout=0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32到64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：64*64 到 32*32

            IC(64, dropout=0.5),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 64到128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：32*32 到 16*16

            IC(128, dropout=0.5),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 128到64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：16x16 到 8x8
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 展平特征图
            nn.Linear(64*8*8, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout 防止过拟合
            nn.Linear(64, num_classes),  # 输出层，分类数=num_classes
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 通过卷积层
        x = self.fc_layers(x)  # 通过全连接层
        return x


class MLP(nn.Module):  # 继承父类
    def __init__(self, input_data, hide_data, output_data):  # 构造函数(方法)
        super(MLP, self).__init__()  # 调用父类nn.Module中的构造函数,进行初始化工作
        self.fc1 = nn.Linear(input_data, hide_data)  # 初始化权重和偏置
        self.reLu = nn.ReLU()
        self.fc2 = nn.Linear(hide_data, output_data)

    def forward(self, x):  # 定义传播方法
        out = self.fc1(x)  # 将输入数据传达到第一个全连接层
        out = self.reLu(out)  # 将match1的结果激活
        out = self.fc2(out)  # 激活后的输入传达到第二个全连接层
        return out  # 返回最后一个out值
