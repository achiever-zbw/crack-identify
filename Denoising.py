import os
import cv2

#待去噪的图片路径
non_denoising_crack_path='./crack-identify/train_images/crack'
non_denoising_noncrack_path='./crack-identify/train_images/non_crack'

#去噪后的图片路径
denoising_crack_path='./crack-identify/Denoising_train_images/crack'
denoising_noncrack_path='./crack-identify/Denoising_train_images/non_crack'

#中值滤波器窗口大小，必须是奇数
kernal_size=5

#对有裂缝的图片进行中值滤波去噪
for name in os.listdir(non_denoising_crack_path):
    #构造图片的完整路径
    crack_img_path=os.path.join(non_denoising_crack_path,name)
    crack_img = cv2.imread(crack_img_path)
    
    #读取图片
    crack_img=cv2.imread(crack_img_path)

    #进行中值滤波,用opencv中的函数
    denoising_crack_img=cv2.medianBlur(crack_img,kernal_size)

    #保存图片到已经去噪的图片所在的文件夹
    denoising_crack_img_path = os.path.join(denoising_crack_path, name)
    cv2.imwrite(denoising_crack_img_path, denoising_crack_img)

#对非裂缝的图片进行中值滤波去噪
for name in os.listdir(non_denoising_noncrack_path):
    #构造图片的完整路径
    non_crack_img_path=os.path.join(non_denoising_noncrack_path,name)
    noncrack_img = cv2.imread(non_crack_img_path)
    
    #读取图片
    noncrack_img=cv2.imread(non_crack_img_path)

    #进行中值滤波,用opencv中的函数
    denoising_noncrack_img=cv2.medianBlur(noncrack_img,kernal_size)

    #保存图片到已经去噪的图片所在的文件夹
    denoising_noncrack_img_path = os.path.join(denoising_noncrack_path, name)
    cv2.imwrite(denoising_noncrack_img_path, denoising_noncrack_img)
