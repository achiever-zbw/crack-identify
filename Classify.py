"""分类
1.筛选验证集图片
2.筛选训练集图片
"""
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

# 修改标签读取部分
def right_labels(label_file):
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            # 确保文件名格式统一
            img_name = line.strip().split()[0]
            if not img_name.endswith('.jpg'):
                img_name += '.jpg'
            label = line.strip().split()[1]
            label_dict[img_name] = label
    return label_dict

if __name__ == "__main__":
    input_labels_path = './val_txt'
    new_verify_images_path = './crack-identify/Right_labels.txt'

    labels_file = "./crack-identify/Right_labels.txt"
    source_dir = "./crack-identify/verify_images"
    output_dir = "./crack-identify/val_final_images"

    filter_single_feature_images(input_labels_path, new_verify_images_path)
    print("筛选完成")

    filter_val_images(labels_file,source_dir,output_dir)
    print("验证集筛选完成")



# 使用示例：

# label_folder = './label_train'
# source_folder = './image_train'
# crack_folder = './crack-identify/train_final_images/crack'
# non_crack_folder = './crack-identify/train_final_images/non_crack'

# filter_single_feature_images(label_folder, source_folder, crack_folder, non_crack_folder)
# print("训练集筛选完成")
