import taichi as ti
import taichi.math as tm
import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math
from PIL import Image

import skimage
from skimage import io,data

from utils import least_sqaure
import queue

ti.init(arch=ti.gpu)

def img_save(np_img,out_name,c_out_name):
    plt.figure(figsize=(np_img.shape[1]//100,np_img.shape[0]//100))
    # plt.figure(figsize=(10,6))
    plt.imshow(np_img, cmap='viridis', aspect='auto',interpolation="nearest")  # 这里使用viridis colormap，也可以选择其他的colormap
    plt.colorbar()  # 添加颜色条
    plt.title('Plane Segmentation')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.savefig(c_out_name)
    
    np_img = np.expand_dims(np_img, axis=2)
    np_img = np.repeat(np_img, repeats=3, axis=2)
    image = Image.fromarray(np_img.astype('uint8'))
    image.save(out_name)



# 并行方式1
# n = 300
# pixels = ti.field(dtype=float, shape=(n * 2, n))

# @ti.kernel
# def func():
#     for i, j in pixels:
#         pixels[i,j] = i + j

# func()
# np_pixels = pixels.to_numpy()
# print(np_pixels)
# img_save(np_pixels, "testname.png", "c_testname.png")



# 并行方式2
# @ti.func
# def is_prime(n: int):
#     result = True
#     for k in range(2, int(n ** 0.5) + 1):
#         if n % k == 0:
#             result = False
#             break
#     return result

# @ti.kernel
# def count_primes(n: int) -> int:
#     count = 0
#     for k in range(2, n):
#         if is_prime(k):
#             count += 1

#     return count

# print(count_primes(1000000))


# 并行方式3(其实就是2)
# n = 3
# # seeds_list可以换成更复杂的数据结构，而非简单的[x,y]坐标
# seeds_list = np.array([[0,0],[1,1],[2,1]])

# # 从np.array转换成ti.field
# ti_seeds_list = ti.field(dtype=ti.i32, shape=(3,2))
# ti_seeds_list.from_numpy(seeds_list)

# voxels = ti.field(dtype=float, shape=(n,n,n))

# @ti.kernel
# def func():
#     # taichi支持这种for i in range()的并行，我们用i的并行巧妙地得到所有seeds patch 的信息。
#     for i in range(0,n):
#         coordinate_x = ti_seeds_list[i,0]
#         coordinate_y = ti_seeds_list[i,1]
#         voxels[i,coordinate_x,coordinate_y] = 5

# func()

# print(voxels)






# 原来的：plane_list: [seg_idx: err,[a,b,d],seg_idx,growed,points,point_nums]
# 以下是成熟的taichi并行框架。

# 有6个平面分割，在每一个平面idx下，有7个信息：idx, error, a, b, d, (x,y)
plane_list = ti.Vector.field(n=7, dtype=ti.f32, shape=(6,1))
plane_list[0,0][0] = 3.0
plane_list[0,0][1] = 3.2
plane_list[2,0][4] = 103.0
print(plane_list)

seed_num = 6
rgb_img = ti.field(dtype=ti.f32, shape=(20,20))
depth_gt_img = ti.field(dtype=ti.f32, shape=(20,20))
objseg_gt_img = ti.field(dtype=ti.f32, shape=(20,20))
plane_seg_img = ti.field(dtype=ti.f32, shape=(20,20,6))
point_cloud_img = ti.field(dtype=ti.f32, shape=(20,20,3))

@ti.func
def recal():
    return 

@ti.kernel
def func():
    # 所有的种子块并行开始生长
    for i in range(0,seed_num):
        # 该种子块的平面编号
        plane_idx = plane_list[i,0][0]
        # 该种子块的一个像素的x、y坐标
        src_pixel_x = int(plane_list[i,0][5])
        src_pixel_y = int(plane_list[i,0][6])
        # 模拟生长的过程：如果下一个像素点的空间点坐标能拟合上当前平面，就更新相应信息
        if point_cloud_img[src_pixel_x+1, src_pixel_y+1, 0] == 0 :
            # 让该像素的平面编号等于当前平面
            plane_seg_img[src_pixel_x+1, src_pixel_y+1, i] = plane_seg_img[src_pixel_x, src_pixel_y, i] + 13.14
            # 重新计算拟合误差等信息
            recal()
            # 更新相应信息
            plane_list[i,0][1] = 5.21
        p = [i, 2, 3]
        for j in range(3):
            p[0] = j + 1
    return 

func()
print(plane_list)
# print("\nplane_seg_img:\n")
# print(plane_seg_img)
