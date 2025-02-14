import torch
from torch import nn, optim
import torch.nn.functional as F

# 定义IC层

class IC(nn.Module):
    """
    定义IC层
    BatchNorm2d层
    Dropout2d层
    """

    def __init__(self, channels, dropout):
        super(IC, self).__init__()
        self.batchnorm = nn.BatchNorm2d(channels)  # 定义 BatchNorm2d 层
        self.dropout = nn.Dropout2d(dropout)      # 定义 Dropout2d 层

    def forward(self, x):
        x = self.batchnorm(x)  # 批归一化
        x = self.dropout(x)    # Dropout
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

class CNN_4_IC(nn.Module):
    """4层CNN模型,包含IC层

    Args:
        num_classes (int): 分类数
    """
    
    def __init__(self, num_classes):
        super(CNN_4_IC, self).__init__()
        self.conv_layers = nn.Sequential(
            # 第一层保持不变
            IC(3, dropout=0.3),
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 128 -> 64

            # 第二层
            IC(48, dropout=0.3),
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 64 -> 32

            # 第三层
            IC(96, dropout=0.5),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 32 -> 16

            # 第四层 
            IC(192, dropout=0.4),
            nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16 -> 8
            
        )

        # 调整全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 8 * 8, 512),  # 增大维度 512->1024
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 通过卷积层
        x = self.fc_layers(x)  # 通过全连接层
        return x

class CNN_4(nn.Module):
    """4层CNN模型

    Args:
        num_classes (int): 分类数
        dropout (float): Dropout率
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充
    """

    def __init__(self, num_classes):
        """初始化
        """
        super(CNN_4, self).__init__()
        self.conv_layers = nn.Sequential(
            # 第一层 - 减少初始通道数
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),  # 48->32
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2),

            # 第二层 - 平缓增长
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),  # 96->64
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2),

            # 第三层 - 适度增长
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),  # 192->128
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout2d(0.6),
            nn.MaxPool2d(kernel_size=2),

            # 第四层 - 平滑收敛
            nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),  # 64->48
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(kernel_size=2)
        )

        # 调整全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 16 * 16, 512),  # 增大维度 512->1024
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """前向传播

        Args:
            x (tensor): 输入数据

        Returns:
            tensor: 输出数据
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
class CNN_4_new(nn.Module):
    """4层CNN模型

    Args:
        num_classes (int): 分类数
        dropout (float): Dropout率
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充
    """

    def __init__(self, num_classes):
        """初始化
        """
        super(CNN_4_new, self).__init__()
        self.conv_layers = nn.Sequential(
            # 第一层 - 减少初始通道数
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),  # 48->32
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2),

            # 第二层 - 平缓增长
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),  # 96->64
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2),

            # 第三层 - 适度增长
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),  # 192->128
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(kernel_size=2),

            # 第四层 - 平滑收敛
            nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),  # 64->48
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2)
        )

        # 调整全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 8 * 8, 512),  # 增大维度 512->1024
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """前向传播
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
class CNN_4_2(nn.Module):
    """4层CNN模型

    Args:
        num_classes (int): 分类数
        dropout (float): Dropout率
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充
    """

    def __init__(self, num_classes):
        """初始化
        """
        super(CNN_4_2, self).__init__()
        self.conv_layers = nn.Sequential(
            # 第一层 - 减少初始通道数
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 48->32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2),

            # 第二层 - 平缓增长
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 96->64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2),

            # 第三层 - 适度增长
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 192->128
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.6),
            nn.MaxPool2d(kernel_size=2),

            # 第四层 - 平滑收敛
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # 64->48
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(kernel_size=2)
        )

        # 调整全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),  # 增大维度 512->1024
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """前向传播

        Args:
            x (tensor): 输入数据

        Returns:
            tensor: 输出数据
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x