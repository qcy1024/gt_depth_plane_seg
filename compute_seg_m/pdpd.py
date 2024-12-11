import os, sys
import math
import queue

from PIL import Image

import skimage
from skimage import io,data

import taichi as ti
import taichi.math as tm

import torch

import numpy as np
import matplotlib.pyplot as plt

from utils import least_sqaure
import plane_seg
import comp

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
    return 


tivec3 = ti.types.vector(3, float)

img_depth_gt_path = "198_depth_gt.png"
img_depth_gt = skimage.io.imread(img_depth_gt_path)
img_depth_gt = img_depth_gt.astype(np.float32)
img_depth_gt_field = ti.field(dtype=ti.f32, shape=(img_depth_gt.shape[0], img_depth_gt.shape[1]))
img_depth_gt_field.from_numpy(img_depth_gt)
img_objseg_gt_name = "198_obj_seg.png"
img_objseg_gt = skimage.io.imread(img_objseg_gt_name)
img_objseg_gt_field = ti.field(dtype=ti.f32, shape=(img_objseg_gt.shape[0], img_objseg_gt.shape[1]))
img_objseg_gt_field.from_numpy(img_objseg_gt)
field_point_cloud = ti.field(dtype=ti.f32, shape=(img_depth_gt.shape[0], img_depth_gt.shape[1], 3))
dd = 1
img_depth_gt /= dd
L = 10

# inv_K = torch.tensor( [ [5.8262448167737955e+02,0.000000e+00,3.1304475870804731e+02],
#                         [0.000000e+00,5.8269103270988637e+02,2.3844389626620386e+02],
#                         [0.000000e+00,0.000000e+00,1.000000e+00] ])
# inv_K_T = torch.inverse(inv_K)
inv_K = np.array([[5.8262448167737955e+02,0.000000e+00,3.1304475870804731e+02],
                    [0.000000e+00,5.8269103270988637e+02,2.3844389626620386e+02],
                    [0.000000e+00,0.000000e+00,1.000000e+00] ])
inv_K_T = np.linalg.inv(inv_K)
inv_K_T_field = ti.field(dtype=ti.f32, shape=(inv_K_T.shape[0], inv_K_T.shape[1]))
inv_K_T_field.from_numpy(inv_K_T)

inv_K_torch = torch.tensor( [ [5.8262448167737955e+02,0.000000e+00,3.1304475870804731e+02],
                        [0.000000e+00,5.8269103270988637e+02,2.3844389626620386e+02],
                        [0.000000e+00,0.000000e+00,1.000000e+00] ])
inv_K_torch_T = inv_K_torch.inverse()


# 成熟的函数：这是可以的，这个函数测试通过！已经可以并行得到空间点云了！

@ti.kernel
def depth_to_space_point():
    for i, j in img_depth_gt_field:
        p = tivec3([i,j,1])
        D = img_depth_gt_field[i,j]
        # print("D = ", D)
        if D != 0:
            # point_cloud[i,j] = D * torch.mv(inv_K_T,p)
            result = tivec3([0.0,0.0,0.0])
            sum = 0.0
            for k in range(3):
                sum = 0.0
                for l in range(3):
                    sum = sum + inv_K_T_field[k,l] * p[l]
                # print("sum = ", sum)
                result[k] = sum
                field_point_cloud[i,j,k] = D * result[k]
    return 

if __name__ == "__main__" :
    print("inv_K_T = ",inv_K_T)

    # # depth_to_space_point()测试通过！
    # depth_to_space_point()
    # pdpd_cloud = point_cloud_img.to_numpy()
    # dpd_cloud = plane_seg.depth_to_space_point(img_depth_gt, inv_K_torch_T)
    # dpd_cloud = dpd_cloud.numpy()
    
    # if comp.comp_img(dpd_cloud, pdpd_cloud):
    #     print("Depth to space point test: True")
    # else :
    #     print("Depth to space point test: false")


    print()
    pass



      
