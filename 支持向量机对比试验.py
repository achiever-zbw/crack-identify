"""支持向量机对比试验
1.数据集划分
2.数据预处理
3.模型创建
4.训练模型
5.验证模型
"""
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from Classify import right_labels

# 数据路径
data_path = './crack-identify/Denoising_train_images'

transform = transforms.Compose([
    transforms.Resize([128, 128]),  # 输入调整为128x128
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
])

# 加载训练数据集
dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 提取训练数据和标签
data = []
labels = []

for inputs, label in dataloader:
    # 将图像展平为一维向量
    data.append(inputs.view(inputs.size(0), -1).numpy())  # 展平
    labels.append(label.numpy())  # 标签

# 合并所有数据
data = np.concatenate(data, axis=0)
labels = np.concatenate(labels, axis=0)

# 数据标准化
scaler = StandardScaler()
data = scaler.fit_transform(data)

# 训练SVM模型
svm_model = SVC(kernel='linear')  # 线性核的SVM
svm_model.fit(data, labels)

print("SVM模型训练完毕.")

# ----------------------验证阶段---------------------------------------------------------

# 定义路径
Right_labels_path = './crack-identify/Right_labels.txt'
right_label_dict = right_labels(Right_labels_path)

# 定义路径

denoising_verify_images_path = './crack-identify/Denoising_verify_images'

# 获取去噪后的验证图片路径
verify_images = [os.path.join(denoising_verify_images_path, f)
                 for f in os.listdir(denoising_verify_images_path) ]

# 获取验证集标签
verify_labels = []
for img_path in verify_images:
    img_name = os.path.basename(img_path)
    label = right_label_dict.get(img_name, None)  # 获取标签
    if label == "crack":
        label_new = 0
        verify_labels.append(label_new)
    else:
        label_new = 1
        verify_labels.append(label_new)

# 预测和验证阶段
prediction = []

# 遍历所有验证集图片
for image_path in verify_images:
    img = Image.open(image_path).convert('RGB')  # 打开图像并转换为RGB模式
    img_tensor = transform(img).unsqueeze(0)  # 应用变换并添加批次维度
    
    # 展平图像
    img_data = img_tensor.view(1, -1).numpy()

    # 标准化
    img_data = scaler.transform(img_data)
    
    # 使用SVM模型进行预测
    prediction_label = svm_model.predict(img_data)
    prediction.append(prediction_label[0])  # 预测结果

# 计算准确率
accuracy = accuracy_score(verify_labels, prediction)
print(f'验证集准确率: {accuracy * 100:.2f}%')

