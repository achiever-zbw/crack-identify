"""去噪
1.对训练集进行中值滤波去噪
2.对验证集进行中值滤波去噪
"""
import os
import cv2

def medianblur(input_path, output_path, kernel_size):
    # 对指定位置的图片进行中值滤波去噪
    """对训练集进行中值滤波去噪

    Args:
        input_path (str): 待去噪的图片路径
        output_path (str): 去噪后的图片路径
        kernel_size (int): 中值滤波器窗口大小，必须是奇数
    """
    for name in os.listdir(input_path):
        input_img_path = os.path.join(input_path, name)
        output_img_path = os.path.join(output_path, name)

        # 读取图片
        img = cv2.imread(input_img_path)

        # 进行中值滤波
        denoised_img = cv2.medianBlur(img, kernel_size)

        # 保存去噪后的图片
        cv2.imwrite(output_img_path, denoised_img)


def medianblur_val(input_path,output_path,kernal_size):
    """对验证集进行中值滤波去噪

    Args:
        input_path (str): 待去噪的图片路径
        output_path (str): 去噪后的图片路径
        kernal_size (int): 中值滤波器窗口大小，必须是奇数
    """
    for img_name in os.listdir(input_path):
        input_image_path = os.path.join(input_path, img_name)
        output_image_path = os.path.join(output_path, img_name)
        # 读取图片
        img = cv2.imread(input_image_path)
        # 进行中值滤波
        denoised_img = cv2.medianBlur(img, kernal_size)
        # 保存去噪后的图片
        cv2.imwrite(output_image_path, denoised_img)
        
if __name__ == "__main__":
    # 待去噪的图片路径
    non_denoising_crack_path = './crack-identify/train_final_images/crack'
    non_denoising_noncrack_path = './crack-identify/train_final_images/non_crack'

# 去噪后的图片路径
    denoising_crack_path = './crack-identify/Denoising_train_images/crack'
    denoising_noncrack_path = './crack-identify/Denoising_train_images/non_crack'

    non_denoising_val_path = './crack-identify/val_final_images'
    denoising_val_path = './crack-identify/Denoising_verify_images'

# 中值滤波器窗口大小，必须是奇数
    kernal_size = 5
    #medianblur(non_denoising_noncrack_path, denoising_noncrack_path, kernal_size)
    #medianblur(non_denoising_crack_path, denoising_crack_path, kernal_size)
    medianblur_val(non_denoising_val_path, denoising_val_path, kernal_size)
    print("图片去噪完成")
