# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:20:46 2024

@author: 赵博文
"""
import os
import os
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


"""
crack_path='./dataset/crack.txt'
with open(crack_path,'r') as f:
    crack_image=[line.strip() for line in f.readlines()]
    
non_crack_path='./dataset/non_crack.txt'
with open(non_crack_path,'r') as f:
    non_crack_image=[line.strip() for line in f.readlines()]
    
train_image_path='./dataset/images/train'
train_crack_image_path='./dataset/train_crack_image'
train_noncrack_image_path='./dataset/train_noncrack_image'

for crack_image_name in crack_image:
    source_path = os.path.join(train_image_path,crack_image_name )  # 构造源路径
    destination_path = os.path.join(train_crack_image_path, crack_image_name)  # 构造目标路径
    
    if os.path.exists(source_path):
       shutil.copy(source_path, destination_path)  # 复制文件到目标文件夹
       print(f"Copied { crack_image_name} to {train_crack_image_path}")
    else:
       print(f"Image {crack_image_name} not found in {train_image_path}")
       
for non_crack_image_name in non_crack_image:
    source_path = os.path.join(train_image_path,non_crack_image_name )  # 构造源路径
    destination_path = os.path.join(train_noncrack_image_path, non_crack_image_name)  # 构造目标路径
    
    if os.path.exists(source_path):
       shutil.copy(source_path, destination_path)  # 复制文件到目标文件夹
       print(f"Copied { non_crack_image_name} to {train_noncrack_image_path}")
    else:
       print(f"Image {non_crack_image_name} not found in {train_image_path}")

"""

data_path='./train_images'

#数据预处理 
transform=transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
    ])

#加载数据集
dataset=datasets.ImageFolder(root=data_path,transform=transform)
dataloader=DataLoader(dataset,32,shuffle=True)

#获取类别
classes=dataset.classes
sum_classes=len(classes)
#print(f"类别：{classes}")

#构建CNN模型
"""
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            #卷积层
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输入通道=3，输出通道=32 ,卷积核大小3*3，步长为1，填充为1
            nn.ReLU(),   # 激活函数
            
            #池化操作(最大池化)
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 尺寸减半：256*256 到 128*128（通过2*2的池化窗口实现）
            nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1), # 输入通道=32保持不变
            #nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)  #32->64
            nn.ReLU(),                                             # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 尺寸减半：128*128 到 64*64
        )
        #全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),     # 展平特征图
            nn.Linear(32 * 64 * 64, 128),     # 输入维度=64*64*64，输出=128
            nn.ReLU(),                                             # 激活函数
            nn.Linear(128, num_classes),                          # 输出层，分类数=num_classes
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 通过卷积层
        x = self.fc_layers(x)    # 通过全连接层
        return x
    
num_classes = len(classes)
model = CNN(num_classes=num_classes)
"""
#构建CNN模型
#卷积层
con_layers=nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), 
    #32*256*256
    # 输入通道=3(颜色)，输出通道=32（自定义，自调整） ,卷积核大小3*3，步长为1，填充为1
    nn.ReLU(),   # 激活函数,增加非线性特征
    
    #池化操作(最大池化)
    nn.MaxPool2d(kernel_size=2, stride=2),                 # 尺寸减半：256*256 到 128*128（通过2*2的池化窗口实现）
    nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1), # 输入通道=32保持不变
    #nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)  #32->64
    nn.ReLU(),                                             # 激活函数
    nn.MaxPool2d(kernel_size=2, stride=2),                 # 尺寸减半：128*128 到 64*64
    )
#全连接层
fc_layers=nn.Sequential(
    nn.Flatten(),     # 展平特征图
    nn.Linear(32 * 64 * 64, 128),     # 输入维度=64*64*64，输出=128
    nn.ReLU(),                                             # 激活函数
    nn.Linear(128, sum_classes)
    )
#定义传播函数
def spread(i):
    i=con_layers(i)
    i=fc_layers(i)
    return i

# 定义损失函数和优化器
loss = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(list(con_layers.parameters())+list(fc_layers.parameters()), lr=0.001)


#训练模型
num_epochs = 15
for each in range(num_epochs):
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    
    #model.train()  # 设置模型为训练模式

    for i, (inputs, labels) in enumerate(dataloader, 0):
        # 将数据送到设备
        
        
        optimizer.zero_grad()  # 梯度清零, 每次训练前防止梯度叠加
        outputs = spread(inputs)  # 向前传播
        loss_data = loss(outputs, labels)  # 计算损失
        
        loss_data.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数
        
        # 累加损失
        running_loss += loss_data.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)  # 获取最大值索引作为预测结果
        total += labels.size(0)  # 总样本数
        correct += (predicted == labels).sum().item()  # 正确的预测数

    # 每个 epoch 的损失和准确率
    each_loss = running_loss / len(dataloader)
    each_acc = 100 * correct / total  # 计算准确率（百分比）
    print(f'Epoch [{each+1}/{num_epochs}], Loss: {each_loss:.2f}, Accuracy: {each_acc:.2f}%')

# 保存模型
torch.save({
    'con_layers': con_layers.state_dict(),
    'fc_layers': fc_layers.state_dict()
}, "cnn_model.pth")
print("模型已保存为 cnn_model.pth")















