from PIL import Image

import skimage
from skimage import io,data
import numpy as np

import os

# 定义目录路径
source_dir = "./nyu_v2_obj_gt"  # 源目录
target_dir = "./raw_obj_seg"  # 目标目录

def img_inverse_rgb(img_rgb):
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1] // 2):
            for k in range(3):
                tmp = img_rgb[i, j, k]
                img_rgb[i, j, k] = img_rgb[i, img_rgb.shape[1]-j-1, k]
                img_rgb[i, img_rgb.shape[1]-j-1, k] = tmp
    return img_rgb

def img_inverse_gray(img_gray):
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1] // 2):
            tmp = img_gray[i, j]
            img_gray[i, j] = img_gray[i, img_gray.shape[1]-j-1]
            img_gray[i, img_gray.shape[1]-j-1] = tmp
    return img_gray

for filename in os.listdir(source_dir):
    if filename.endswith(".png"):  # 检查是否是 PNG 文件
        file_path = os.path.join(source_dir, filename)  # 拼接完整路径
        print(f"正在处理：{file_path}")
        img = skimage.io.imread(file_path)
        img = img.astype(np.float32)
        img = img_inverse_gray(img)
        image = Image.fromarray(img.astype('uint8'))
        target_path = os.path.join(target_dir, filename)
        image.save(target_path)
        print(f"已保存到：{target_path}")


