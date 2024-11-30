import os
import cv2

def medianblur(input_path,output_path,kernel_size):
    """
    #待去噪的图片路径
    non_denoising_crack_path='./crack-identify/train_images/crack'
    non_denoising_noncrack_path='./crack-identify/train_images/non_crack'

    #去噪后的图片路径
    denoising_crack_path='./crack-identify/Denoising_train_images/crack'
    denoising_noncrack_path='./crack-identify/Denoising_train_images/non_crack'

    #中值滤波器窗口大小，必须是奇数
    kernal_size=5
    """
    #对指定位置的图片进行中值滤波去噪
    for name in os.listdir(input_path):
        input_img_path = os.path.join(input_path, name)
        output_img_path = os.path.join(output_path, name)

        # 读取图片
        img = cv2.imread(input_img_path)

        # 进行中值滤波
        denoised_img = cv2.medianBlur(img, kernel_size)

        # 保存去噪后的图片
        cv2.imwrite(output_img_path, denoised_img)
        
