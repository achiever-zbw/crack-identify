

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image  # 用于图像显示
from Denoising import medianblur  # 调用去噪库中的中值滤波函数
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端


# 待去噪的图片路径
non_denoising_crack_path = './crack-identify/train_images/crack'
non_denoising_noncrack_path = './crack-identify/train_images/non_crack'

# 去噪后的图片路径
denoising_crack_path = './crack-identify/Denoising_train_images/crack'
denoising_noncrack_path = './crack-identify/Denoising_train_images/non_crack'

# 中值滤波器窗口大小，必须是奇数
kernal_size = 5
medianblur(non_denoising_noncrack_path, denoising_noncrack_path, kernal_size)
medianblur(non_denoising_crack_path, denoising_crack_path, kernal_size)

data_path = './crack-identify/Denoising_train_images'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize([128, 128]),  # 输入调整为128x128
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
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

# 构建CNN模型


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 输入通道=3，输出通道=32 ,卷积核大小5*5，步长为1，填充为2
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),   # 激活函数

            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：128*128 到 64*64

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32到64
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：64*64 到 32*32

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 64到128
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半：32*32 到 16*16
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 展平特征图
            nn.Linear(128 * 16 * 16, 512),  # 输入维度=128*16*16，输出=512
            nn.ReLU(),  # 激活函数
            nn.Linear(512, num_classes),  # 输出层，分类数=num_classes
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 通过卷积层
        x = self.fc_layers(x)  # 通过全连接层
        return x


# 获取类别数量并初始化模型
num_classes = len(classes)
model = CNN(num_classes=num_classes)

# 定义损失函数和优化器
loss = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        optimizer.zero_grad()  # 梯度清零, 每次训练前防止梯度叠加
        outputs = model(inputs)  # 向前传播
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
    losses.append(each_loss)  # 将当前的损失添加到 losses 列表
    each_acc = 100 * correct / total  # 计算准确率（百分比）
    print(
        f'Epoch [{each+1}/{num_epochs}], Loss: {each_loss:.2f}, Accuracy: {each_acc:.2f}%')


# 绘制损失变化图像并保存
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')  # 绘制损失曲线
plt.xlabel('Epochs')  # X 轴标签
plt.ylabel('Loss')  # Y 轴标签
plt.title('Training Loss over Epochs')  # 图像标题
plt.legend()

# 保存图像到文件
plt.savefig('/crack-identify/loss_curve.png')  # 将图像保存为 .png 文件
print("损失曲线图已保存")
