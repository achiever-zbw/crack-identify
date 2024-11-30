# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:43:04 2024

@author: 赵博文
"""
import cv2
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
from torch.utils.data import DataLoader
from CNN_model import con_layers,fc_layers,spread
from Denoising import medianblur #调用去噪库中的中值滤波函数

#对验证集图片进行去噪
verify_images_path='./crack-identify/verify_images'
denoising_verify_images_path='./crack-identify/Denoising_verify_images'
kernel_size=5
medianblur(verify_images_path,denoising_verify_images_path,kernel_size)

#数据预处理
transform=transforms.Compose([
    transforms.Resize([256, 256]),  # 与训练集图片大小一致
    transforms.ToTensor(),    
    transforms.Normalize([0.5], [0.5])  # 标准化
    ])

#加载验证集
new_verify_images_path=[os.path.join(denoising_verify_images_path, f) for f in os.listdir(denoising_verify_images_path)]  
#os.listdir(n) 把n里的文件名称提取出来并存到一个列表
#os.path.join(a,b)：把a和b的路径合到一起 ,比如'./verity_images/name1.jpg'，方便对每张图片的操作

model=torch.load("cnn_model.pth")

con_layers.load_state_dict(model["con_layers"])
fc_layers.load_state_dict(model["fc_layers"])

con_layers.eval()
fc_layers.eval()

final_path='./crack-identify/final'

predictions=[]
for img_path in new_verify_images_path:
        # 使用cv2读取图片
        img = cv2.imread(img_path)
        img_pic = Image.fromarray(img)   #保留picture形式，方便之后添加文字用
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
       
        # 应用转换并将图像添加批次维度
        img_tensor = transform(img_pic).unsqueeze(0)
        
        # 进行预测
        output = spread(img_tensor)
        _, predicted = torch.max(output, 1)  # 获取预测类别
        if predicted.item()==0:
            label="crack"
        else:
            label="non_crack"
        predictions.append((img_path,label))
        
        draw = ImageDraw.Draw(img_pic)
        #font = ImageFont.load_default()  # 默认字体
        font = ImageFont.truetype("arial.ttf", 40)  # 设置字体大小为40
        word=f"predict class:{label}"
        draw.text((10,10),word,fill="black",font=font)
        img_name = os.path.basename(img_path)  # 提取文件名
        img_pic.save(os.path.join(final_path,img_name))

print("End")
# 打印结果
#for img_path, pred in predictions:
  #  print(f"image: {img_path}, predicted class: {pred}")

