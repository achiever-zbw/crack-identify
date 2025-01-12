import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
from Model import CNN_4,CNN_4_IC  # 导入CNN模型
import torch
import torch.nn as nn 
from Classify import right_labels

# 定义计算损失和返回输出的函数
def model_out_loss(input_data, model):

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不用梯度计算
        output = model(input_data)  # 获取模型的输出
    return output


def model_test(model, test_data):

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model_out_loss(test_data, model)  # 获取模型输出
    return output





# 定义路径
Right_labels_path = './crack-identify/Right_labels.txt'
right_label_dict = right_labels(Right_labels_path)

# 定义路径
# verify_images_path = './crack-identify/verify_images'
denoising_verify_images_path = './crack-identify/Denoising_verify_images'

transform = transforms.Compose([
    transforms.Resize([128, 128]),  # 与训练集图片大小一致
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 标准化
])

# 加载验证集图片路径
new_verify_images_path = [os.path.join(
    denoising_verify_images_path, f) for f in os.listdir(denoising_verify_images_path)]

# 加载模型
model = CNN_4(num_classes=2)  # 创建模型架构
model.load_state_dict(torch.load('trained_model_CNN_4.pth'))  # 加载权重
model.eval()  # 切换到评估模式
# 加载完整的模型（包括结构和权重）

predictions = []
all_images = 0
corrcet_images = 0
for img_path in new_verify_images_path:
    img_name = os.path.basename(img_path)  # 获取图片文件名
    all_images += 1
    # 读取图片
    img = cv2.imread(img_path)
    img_pic = Image.fromarray(img)  # 转为 PIL 图片格式，方便后续处理

    # 图像预处理
    img_tensor = transform(img_pic).unsqueeze(0)  # 增加批次维度（batch size）

    # 将数据传递到 model_test 函数进行预测
    output = model_test(model, img_tensor)  # 获取模型输出

    # 获取预测结果
    _, predicted = torch.max(output, 1)  # 获取预测的类别
    label = "crack" if predicted.item() == 0 else "non_crack"  # 根据预测的类别索引获取标签
    right_label = right_label_dict.get(img_name, "none")
    if label == right_label:
        corrcet_images += 1
    predictions.append((img_name, label))

    # 在图片上添加预测结果
    #
    # 保存带有标签的图片
    # img_name = os.path.basename(img_path)  # 获取图片文件名
    # img_pic.save(os.path.join(final_path, img_name))  # 保存图片到 final 文件夹

print("all_images:", all_images)
print("corrcet_images:", corrcet_images)
accuracy = corrcet_images/all_images*100
print(f"Accuracy: {accuracy:.2f}%")
print("ok")
