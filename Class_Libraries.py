"""类库
用于深度学习用到的类
包含：
    1.数据集处理
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, Dataset
from Model import CNN_4,CNN_4_new
from PIL import Image
class UnbalancedDataset(Dataset):
    """处理不平衡数据集

    Args:
        Dataset (): 数据集
    方法：
        不同程度的数据增强
    """
    def __init__(self,data_dir,is_train=True):
        """初始化

        Args:
            data_dir (str): 数据集路径
            transform (transforms.Compose): 数据增强
            
        """
        self.dataset=datasets.ImageFolder(root=data_dir)
        self.is_train = is_train
        
        # 对少量数据进行数据增强
        self.strong_transform=transforms.Compose([
            transforms.Resize([256, 256]),  # 输入调整为128x128
            transforms.RandomRotation(degrees=15),  # 随机旋转
            transforms.RandomHorizontalFlip(),  # 水平翻转
            transforms.RandomVerticalFlip(),  # 垂直翻转
            transforms.ColorJitter( 
                brightness=0.3, #亮度
                contrast=0.3, #对比度
                saturation=0.3 #饱和度
            ),
            transforms.RandomAutocontrast(p=0.3),  # 自动对比度
            transforms.RandomEqualize(p=0.2),      # 直方图均衡化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 图像标准化
        ])
    
        self.weak_transform=transforms.Compose([
            transforms.Resize([256, 256]),  # 输入调整为128x128
            transforms.RandomRotation(degrees=15),  # 随机旋转
            transforms.RandomHorizontalFlip(),  # 水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 图像标准化
            
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    
    def __getitem__(self,index):
        """根据索引进行不同类型的数据增强，对上述的实体化操作
           其中，验证集不进行数据增强 

        Args:
            index (int): 索引
        Returns:
            img (PIL.Image): 图像
            label (int): 标签
        """
        img,label=self.dataset[index]
        if not self.is_train:  # 验证集
            return self.val_transform(img), label
        
        if label==0:
            img=self.weak_transform(img)
        else:
            img=self.strong_transform(img)
        return img,label
    
    def __len__(self):
        """返回数据集长度

        Returns:
            int: 数据集长度
        """
        return len(self.dataset)
    
