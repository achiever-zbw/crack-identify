import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from Denoising import medianblur  # 调用去噪库中的中值滤波函数
from Verify import right_labels

# 数据路径
data_path = './crack-identify/Denoising_train_images'

# 数据预处理：将图像调整为128x128并转换为Tensor
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
verify_images_path = './crack-identify/verify_images'
denoising_verify_images_path = './crack-identify/Denoising_verify_images'
final_path = './crack-identify/final'
kernel_size = 5  # 中值滤波器窗口大小

# 去噪验证集图片
medianblur(verify_images_path, denoising_verify_images_path, kernel_size)

# 获取去噪后的验证图片路径
verify_images = [os.path.join(denoising_verify_images_path, f)
                 for f in os.listdir(denoising_verify_images_path) if f.endswith('.jpg') or f.endswith('.png')]

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

# 保存结果到文件
with open(os.path.join(final_path, 'svm_verification_results.txt'), 'w') as f:
    for img_path, pred, true_label in zip(verify_images, prediction, verify_labels):
        f.write(f'{img_path}: Predicted: {pred}, True label: {true_label}\n')

print("SVM验证结果已保存至svm_verification_results.txt")
