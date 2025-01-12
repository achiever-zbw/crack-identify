import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 图像标准化
])

dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 提取数据和标签
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

# 训练朴素贝叶斯模型
nb_model = GaussianNB()
nb_model.fit(data, labels)

print("朴素贝叶斯模型训练完毕.")


# ----------------------验证阶段---------------------------------------------------------


# 定义路径
Right_labels_path = './crack-identify/Right_labels.txt'
right_label_dict = right_labels(Right_labels_path)

# 定义路径

denoising_verify_images_path = './crack-identify/Denoising_verify_images'

verify_images = [os.path.join(denoising_verify_images_path, f)
                 for f in os.listdir(denoising_verify_images_path)]


# 获取验证集真实标签
verify_true_labels = []
for img_path in verify_images:
    img_name = os.path.basename(img_path)
    label = right_label_dict[img_name]  # 获取标签
    if label == "crack":
        label_new = 0
        verify_true_labels.append(label_new)
    else:
        label_new = 1
        verify_true_labels.append(label_new)

prediction = []
for image_path in verify_images:
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    # 展平
    img_data = img_tensor.view(1, -1).numpy()
    # 标准化
    img_data = scaler.transform(img_data)
    # 使用朴素贝叶斯模型进行预测
    prediction_label = nb_model.predict(img_data)
    prediction.append(prediction_label[0])  # 预测结果

accuracy = accuracy_score(verify_true_labels, prediction)
print(f'朴素贝叶斯验证集准确率: {accuracy * 100:.2f}%')

# 检查训练数据的标签分布
train_path = './crack-identify/Denoising_train_images'
crack_path = os.path.join(train_path, 'crack')
noncrack_path = os.path.join(train_path, 'non_crack')

print("训练数据标签分布：")
print(f"crack文件夹（标签0）图片数量: {len(os.listdir(crack_path))}")
print(f"non_crack文件夹（标签1）图片数量: {len(os.listdir(noncrack_path))}")
