import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from CNN_model import CNN_feature_take

transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

data_path = './crack-identify/Denoising_train_images'
dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = CNN_feature_take(num_classes=2)
model.eval()

# 提取特征，准备用于随机森林训练


# 提取特征并展平图像
def features_take(dataloader):
    features = []
    labels = []
    for inputs, label in dataloader:
        # 将每张图像展平为一维向量
        inputs = inputs.view(inputs.size(0), -1)
        features.append(inputs.numpy())  # 将特征转换为NumPy数组
        labels.append(label.numpy())  # 将标签转换为NumPy数组

    features = np.vstack(features)  # 将所有的特征堆叠在一起，形成一个大数组
    labels = np.hstack(labels)  # 将所有的标签堆叠在一起
    return features, labels


features, labels = features_take(dataloader)
# print(labels)
# print(features)
# print(len(features))
# print(len(labels))
# # 使用随机森林进行训练
# X_train, X_test, y_train, y_test = train_test_split(
#     features, labels, test_size=0.2, random_state=42)
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)
# y_pred = rf_classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'随机森林训练准确率: {accuracy * 100:.2f}%')
