import taichi as ti
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

ti.init(arch=ti.cpu)

# 将img_depth_gt的像素坐标变到相机坐标，得到每一个像素点对应的空间点坐标，返回整张图对应的所有空间点坐标
def depth_to_space_point(img_depth_gt,inv_K_T):
    point_cloud = torch.zeros((img_depth_gt.shape[0],img_depth_gt.shape[1],3))
    # print(point_cloud.shape)
    for i in range(img_depth_gt.shape[0]):
        for j in range(img_depth_gt.shape[1]):
            p = torch.tensor([i,j,1],dtype=torch.float32)
            D = img_depth_gt[i,j]
            if D != 0:
                point_cloud[i,j] = D * torch.mv(inv_K_T,p)
                # print(point_cloud[i,j])
    return point_cloud

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


def get_seed_patch_seg(img_depth_gt,L,point_cloud,img_obj_seg_gt):
    # 拟合误差，平面方程[a,b,d]，平面编号，第一个像素的坐标，是否生长好了
    seed_list = {}
    seed_seg = torch.zeros((img_depth_gt.shape[0],img_depth_gt.shape[1]),dtype=torch.int32)
    i_idx = 0
    j_idx = 0
    seg_idx = 1
    while i_idx <= img_depth_gt.shape[0]-L:
        while j_idx <= img_depth_gt.shape[1]-L:
            can_be_seed_patch = True
            obj_seg = img_obj_seg_gt[i_idx][j_idx]
            for k in range(L):
                if k >= img_depth_gt.shape[0]:
                    break
                for l in range(L):
                    if l >= img_depth_gt.shape[1]:
                        break
                    if img_depth_gt[i_idx+k][j_idx+l] == 0 or seed_seg[i_idx+k][j_idx+l] != 0 or img_obj_seg_gt[i_idx+k][j_idx+l] != img_obj_seg_gt[i_idx][j_idx]:
                        can_be_seed_patch = False
                        break 
                if can_be_seed_patch == False:
                    break
            if can_be_seed_patch:
                # print("seed patch found. seg_idx = ",seg_idx)
                x_list = []
                y_list = []
                z_list = []
                points = []
                for k in range(L):
                    if k >= img_depth_gt.shape[0]:
                        break
                    for l in range(L):    
                        if l >= img_depth_gt.shape[1]:
                            break
                        seed_seg[i_idx+k,j_idx+l] = seg_idx
                        # print("seed_seg[", i_idx+k,"]","[",j_idx+l,"] = ",seg_idx)
                        x_list.append(point_cloud[i_idx+k,j_idx+l,0])
                        y_list.append(point_cloud[i_idx+k,j_idx+l,1])
                        z_list.append(point_cloud[i_idx+k,j_idx+l,2])
                        
                        # print("i_idx+k=",i_idx+k,", j_idx+l=",j_idx+l,end=" ")
                        points.append([i_idx+k,j_idx+l])
                # print("seg_idx=",seg_idx)
                x_list = np.array(x_list)
                y_list = np.array(y_list)
                z_list = np.array(z_list)
                N = L * L
                X = least_sqaure(N,x_list,y_list,z_list)
                # print('now seg_idx=',seg_idx,'平面拟合结果为：z = %.3f * x + %.3f * y + %.3f' % (X[0], X[1], X[2]))
                # X[0]: a, X[1]: b, X[2]: d
                a = X[0].item()
                # print("a=",a,"type(a)=",type(a))
                b = X[1].item()
                d = X[2].item()
                # calculate err 
                err = 0
                sum = 0
                for k in range(L):
                    if k >= img_depth_gt.shape[0]:
                        break
                    for l in range(L):    
                        if l >= img_depth_gt.shape[1]:
                            break
                        normal_n = torch.tensor([a,b,-1])
                        p = point_cloud[i_idx+k,j_idx+l]
                        n_dot_p_plus_d = torch.dot(normal_n,p) + d
                        sum = sum + n_dot_p_plus_d * n_dot_p_plus_d
                        # print("normal_n = ",normal_n,"p = ",p,"d = ",d,"\ntorch.dot(normal_n,p) = ",torch.dot(normal_n,p),
                        #    "torch.dot(normal_n,p) + d = ",torch.dot(normal_n,p)+d,"sum = ",sum)
                sum = sum.item()
                sum = sum / L / L
                # print("sum = ",sum,"type(sum) = ",type(sum))
                if sum > 0:
                    sum = math.sqrt(sum)
                err = sum 
                # print("err = ",err)
                
                # initialize seed_list as plane_list
                # print("seg_idx = ",seg_idx)
                seed_list[seg_idx] = [err,[a,b,d],seg_idx,False,points,L*L]
                seg_idx += 1
                j_idx += L  
            # end if can_be_seed_patch
            else :
                j_idx += 1
        # end  while j_idx <= img_depth_gt.shape[1]-L:
        j_idx = 0
        i_idx += 1
    print("seed_list calculated. ")
    return seed_seg, seg_idx-1, seed_list


img_depth_gt_path = "25_depth_gt.png"
img_depth_gt = skimage.io.imread(img_depth_gt_path)
# img_depth_gt.shap=(375,1242), img_depth_gt.type=numpy.ndarray img.shape: (375,1242,3)
# print("img_depth_gt.shape=",img_depth_gt.shape,"img_depth_gt.type=",type(img_depth_gt))     
img_depth_gt = img_depth_gt.astype(np.float32)
ti_img_depth_gt = ti.field(dtype=float,shape=img_depth_gt.shape)
ti_img_depth_gt.from_numpy(img_depth_gt)
# print("img_depth_gt = ",img_depth_gt)
# print("ti_img_depth_gt = ",ti_img_depth_gt)


img_obj_seg_gt_name = "25_obj_seg.png"
img_obj_seg_gt = skimage.io.imread(img_obj_seg_gt_name)
ti_obj_seg_gt = ti.field(dtype=float,shape=img_depth_gt.shape)
ti_obj_seg_gt.from_numpy(img_obj_seg_gt)

inv_K = torch.tensor( [ [5.8262448167737955e+02,0.000000e+00,3.1304475870804731e+02],
                        [0.000000e+00,5.8269103270988637e+02,2.3844389626620386e+02],
                        [0.000000e+00,0.000000e+00,1.000000e+00] ] )
inv_K_T = torch.inverse(inv_K)
print("inv_K_T = ",inv_K_T)
L = 3


# point_cloud = depth_to_space_point(img_depth_gt,inv_K_T)
# init_plane_seg, init_seg_cnt, init_seed_list = get_seed_patch_seg(img_depth_gt,L,point_cloud,img_obj_seg_gt)
# ti_init_seg_cnt = ti.field(dtype=int,shape=init_seg_cnt)

# print(init_seed_list[1])

# 可以
# for i in range(init_seg_cnt):
#     print(init_seed_list[i+1])
    
a = np.array([[1.0,[1.1,1.2,1.3],1,False,[[1,2],[3,4]],2],[1.0,[1.1,1.2,1.3],1,False,[[1,2],[3,4]],2]])

print(a)    # 可以
print(type(a))  # class 'numpy.ndarray'

Len_a = len(a)
print(a[0])
print(a[1])

@ti.kernel
def compute_seg():
    for i in ti.ndrange(Len_a):
        print(i)      # 可以
        print(a[i])     # 不行
        continue
    return 

compute_seg()