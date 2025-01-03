
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import itertools
import sys
from Model import CNN_3
import torch
from torch.optim import lr_scheduler
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image  # 用于图像显示
# from Denoising import medianblur  # 调用去噪库中的中值滤波函数

import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端


data_path = './crack-identify/Denoising_train_images'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize([128, 128]),  # 输入调整为128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 图像标准化
])

# 加载数据集
dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, 32, shuffle=True)

# 获取类别
classes = dataset.classes
sum_classes = len(classes)


# 获取类别数量并初始化模型
num_classes = len(classes)
model = CNN_3(num_classes=num_classes)
# 定义损失函数和优化器
loss = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.002)
scheduler = lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5)  # 每20个epoch将学习率减半
# 记录每个 epoch 的损失
losses = []

# 训练模型
num_epochs = 200
for each in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0.0
    total = 0.0

    for i, (inputs, labels) in enumerate(dataloader, 0):
        optimizer.zero_grad()  # 梯度清零

        outputs = model(inputs)  # 前向传播
        loss_data = loss(outputs, labels)  # 计算损失
        loss_data.backward()  # 反向传播

        optimizer.step()  # 优化器更新参数

        running_loss += loss_data.item()  # 累加损失

        # 计算准确率
        _, predicted = torch.max(outputs, 1)  # 获取最大值索引作为预测结果
        total += labels.size(0)  # 总样本数
        correct += (predicted == labels).sum().item()  # 正确的预测数

     # 计算每个 epoch 的损失和准确率
    each_loss = running_loss / len(dataloader)
    losses.append(each_loss)  # 将当前的损失添加到 losses 列表
    each_acc = 100 * correct / total  # 计算准确率
    print(
        f'Epoch [{each+1}/{num_epochs}], Loss: {each_loss:.4f}, Accuracy: {each_acc:.4f}%')
    # 动态调整学习率
    scheduler.step()

# 绘制损失变化图像并保存
plt.plot(range(1, 201), losses,
         label='Training Loss')  # 绘制损失曲线
plt.xlabel('Epochs')  # X 轴标签
plt.ylabel('Loss')  # Y 轴标签
plt.title('Training Loss over Epochs')  # 图像标题
plt.legend()
# 保存图像到文件
plt.savefig('12.28_CNN_3_IC.png')  # 将图像保存为 .png 文件
print("损失曲线图已保存")

# 保存模型
torch.save(model.state_dict(), 'trained_model.pth')
print("模型已保存")

