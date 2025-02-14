import numpy as np
import sympy as sp
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from Model import CNN_4,CNN_4_new # CNN_4的模型导入
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Class_Libraries import UnbalancedDataset
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端



data_dir = './crack-identify/Denoising_train_images'
full_dataset = UnbalancedDataset(data_dir=data_dir, is_train=True)
total_size = len(full_dataset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

# 创建验证集时使用is_train=False
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 验证集使用单独的transform
val_dataset.dataset.is_train = False  # 设置验证集不使用数据增强

train_dataloader = DataLoader(train_dataset, 32, shuffle=True)
val_dataloader = DataLoader(val_dataset, 32, shuffle=False)
print(f"数据集大小:")
print(f"总数据集: {total_size}")
print(f"训练集: {train_size}")
print(f"验证集: {val_size}")
# 获取类别

model = CNN_4(num_classes=2)
print("模型已创建")
# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.004,
    weight_decay=0.1,
    betas=(0.9, 0.999)
)

# 学习率调度器
scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.3)
# 记录每个 epoch 的损失
losses = []
# 训练模型
num_epochs = 200
for each in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0

    # 打印当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {each}, Current LR: {current_lr:.6f}')

    for i, (inputs, labels) in enumerate(train_dataloader, 0):
        optimizer.zero_grad()  # 梯度清零

        outputs = model(inputs)  # 前向传播
        loss_data = loss_function(outputs, labels)  # 计算损失
        loss_data.backward()  # 反向传播

        optimizer.step()  # 优化器更新参数
        train_loss += loss_data.item()  # 累加损失

        # 计算准确率
        _, predicted = torch.max(outputs, 1)  # 获取最大值索引作为预测结果
        train_total += labels.size(0)  # 总样本数
        train_correct += (predicted == labels).sum().item()  # 正确的预测数
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs, labels
            outputs = model(inputs)
            loss_data = loss_function(outputs, labels)

            val_loss += loss_data.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    # 计算准确率
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total
    # 计算每个epoch的损失和准确率
    scheduler.step()  # 动态调整学习率
    each_loss = train_loss / len(train_dataloader)
    losses.append(each_loss)
    each_acc = 100 * train_correct / train_total
    print(
        f'Epoch [{each+1}/{num_epochs}], Loss: {each_loss:.4f}, Accuracy: {each_acc:.4f}%')
    print(
        f'Val Loss: {val_loss/len(val_dataloader):.4f} | Val Acc: {val_acc:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'trained_model_CNN_4_图片大小.pth')
print("模型已保存")


# 绘制损失变化图像并保存
plt.plot(range(1, 201), losses,
         label='Training Loss')  # 绘制损失曲线
plt.xlabel('Epochs')  # X 轴标签
plt.ylabel('Loss')  # Y 轴标签
plt.title('Training Loss over Epochs')  # 图像标题
plt.legend()
# 保存图像到文件
plt.savefig('2.4_图片大小.png')  # 将图像保存为 .png 文件
print("损失曲线图已保存")
