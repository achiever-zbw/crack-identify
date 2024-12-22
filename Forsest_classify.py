import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Verify import right_labels
import cv2
import os
from PIL import Image
from Denoising import medianblur

transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

data_path = './crack-identify/Denoising_train_images'
dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 提取特征，准备用于随机森林训练


# 提取特征并展平图像
def features_take_train(dataloader):
    features = []
    labels = []
    for inputs, label in dataloader:
        # 将每张图像展平为一维向量
        inputs = inputs.view(inputs.size(0), -1)
        features.append(inputs.numpy())  # 将特征转换为NumPy数组
        labels.append(label.numpy())  # 将标签转换为NumPy数组

    features = np.vstack(features)   # 将所有的特征堆叠在一起，形成一个大数组
    labels = np.hstack(labels)  # 将所有的标签堆叠在一起
    return features, labels


features_train, labels_train = features_take_train(dataloader)
# print(labels)
# print(features)
print(len(features_train))
print(len(labels_train))

# 使用随机森林进行训练
rf_classify = RandomForestClassifier(n_estimators=100, random_state=42)
# 使用整个训练集进行训练
rf_classify.fit(features_train, labels_train)
# fit(x,y)  x-->特征矩阵，二维的结构  y-->标签，一维数组
print("随机森林训练完毕")


# ----------------------进行验证--------------------------


# 定义路径
Right_labels_path = './crack-identify/Right_labels.txt'
right_label_dict = right_labels(Right_labels_path)

# 定义路径
verify_images_path = './crack-identify/verify_images'
denoising_verify_images_path = './crack-identify/Denoising_verify_images'
final_path = './crack-identify/final'
kernel_size = 5  # 中值滤波器窗口大小

# 去噪验证集图片
medianblur(verify_images_path, denoising_verify_images_path, kernel_size)

verify_images = [os.path.join(denoising_verify_images_path, f)
                 for f in os.listdir(denoising_verify_images_path)]


def features_take_verify(verify_images, transform):
    features = []
    for img_path in verify_images:
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = img_tensor.view(img_tensor.size(0), -1)  # 展平为一维向量
        features.append(img_tensor.numpy())
    features = np.vstack(features)
    return features


features_verify = features_take_verify(verify_images, transform)
print(f"验证集图片数量为{len(features_verify)}")

# 获取验证集标签
verify_labels = []
for img_path in verify_images:
    img_name = os.path.basename(img_path)
    label = right_label_dict[img_name]  # 获取标签
    if label == "crack":
        label_new = 0
        verify_labels.append(label_new)
    else:
        label_new = 1
        verify_labels.append(label_new)
print(verify_labels)
print(len(verify_labels))
# 对验证集进行预测
verify_predictions = rf_classify.predict(features_verify)
# 计算准确率
accuracy = accuracy_score(verify_labels, verify_predictions)
print(f"验证集准确率: {accuracy * 100:.2f}%")