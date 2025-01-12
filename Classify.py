import os
import shutil


def filter_single_feature_images(input_folder, output_file):
    """筛选验证集图片并保存为New_labels格式
    条件：图片中的所有特征必须属于同一类别
    crack: 所有特征都是0/1/2/3
    non_crack: 所有特征都不是0/1/2/3

    Args:
        input_folder (str): 输入标签文件夹路径
        output_file (str): 输出文件路径

    Returns:
        None
    """
    crack_count = 0
    non_crack_count = 0
    
    with open(output_file, "w") as new:
        for name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, name)

            with open(img_path, "r") as f:
                lines = f.readlines()
                
                # 获取所有特征
                features = [int(line.strip().split()[0]) for line in lines]
                
                # 检查是否所有特征都是裂缝类型
                all_crack = all(f in [0, 1, 2, 3] for f in features)
                # 检查是否所有特征都是非裂缝类型
                all_non_crack = all(f not in [0, 1, 2, 3] for f in features)

                # 只有当所有特征都属于同一类别时才保存
                if all_crack:
                    base_name = os.path.splitext(name)[0]
                    jpg_filename = f"{base_name}.jpg"
                    new.write(f"{jpg_filename} crack\n")
                    crack_count += 1
                elif all_non_crack:
                    base_name = os.path.splitext(name)[0]
                    jpg_filename = f"{base_name}.jpg"
                    new.write(f"{jpg_filename} non_crack\n")
                    non_crack_count += 1

    print(f"筛选完成:")
    print(f"裂缝图片数量: {crack_count}")
    print(f"非裂缝图片数量: {non_crack_count}")


def filter_val_images(labels_file, source_dir, output_dir):
    """根据Right_labels.txt筛选验证集图片

    Args:
        labels_file (str): Right_labels.txt的路径
        source_dir (str): verify_images源文件夹路径
        output_dir (str): val_final_images目标文件夹路径
    """
    # 读取Right_labels.txt中的图片名称
    val_images = []
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.isspace():
                continue
            # 去掉行号，只保留图片名
            image_name = line.split()[0].strip()
            val_images.append(image_name)

    # 复制符合条件的图片
    for image_name in val_images:
        correct_path = os.path.join(source_dir, image_name)
        dst_path = os.path.join(output_dir, image_name)
        shutil.copy2(correct_path, dst_path)


def right_labels(Right_labels_path):
    right_label = {}
    with open(Right_labels_path, "r") as f:
        for line in f:
            part = line.strip().split(" ")
            right_label[part[0]] = part[1]

    return right_label


input_labels_path = './val_txt'
new_verify_images_path = './crack-identify/Right_labels.txt'

labels_file = "./crack-identify/Right_labels.txt"
source_dir = "./crack-identify/verify_images"
output_dir = "./crack-identify/val_final_images"

# filter_single_feature_images(input_labels_path, new_verify_images_path)
# print("筛选完成")

# filter_val_images(labels_file,source_dir,output_dir)
# print("验证集筛选完成")


def filter_single_feature_images(label_folder, source_folder, crack_folder, non_crack_folder):
    """筛选单一特征图片并分类保存
    crack: 只有一个特征且是0/1/2/3的图片
    non_crack: 只有一个特征且不是0/1/2/3的图片

    Args:
        label_folder (str): 标签文件夹路径 (label_train)
        source_folder (str): 源图片文件夹路径 (train_images)
        crack_folder (str): 裂缝图片保存路径 (crack)
        non_crack_folder (str): 非裂缝图片保存路径 (non_crack)
    """
    # 确保输出文件夹存在
    os.makedirs(crack_folder, exist_ok=True)
    os.makedirs(non_crack_folder, exist_ok=True)

    crack_count = 0
    non_crack_count = 0

    for label_file in os.listdir(label_folder):
        label_path = os.path.join(label_folder, label_file)

        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()

            # 只处理单一特征的图片
            if len(lines) == 1:
                # 获取唯一特征
                feature = int(lines[0].strip().split()[0])

                # 构建源图片路径
                img_name = os.path.splitext(label_file)[0] + '.jpg'
                source_img_path = os.path.join(source_folder, img_name)

                # 判断类别
                if feature in [0, 1, 2, 3]:
                    # 单一特征是0/1/2/3，复制到裂缝文件夹
                    shutil.copy2(source_img_path, os.path.join(
                        crack_folder, img_name))
                    crack_count += 1
                else:
                    # 单一特征不是0/1/2/3，复制到非裂缝文件夹
                    shutil.copy2(source_img_path, os.path.join(
                        non_crack_folder, img_name))
                    non_crack_count += 1
    
    
# 使用示例：

# label_folder = './label_train'
# source_folder = './image_train'
# crack_folder = './crack-identify/train_final_images/crack'
# non_crack_folder = './crack-identify/train_final_images/non_crack'

# filter_single_feature_images(label_folder, source_folder, crack_folder, non_crack_folder)
# print("训练集筛选完成")
