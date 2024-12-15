
import os, sys
import math
import queue
import time

from PIL import Image

import skimage
from skimage import io,data

import taichi as ti
import taichi.math as tm

import torch

import numpy as np
import matplotlib.pyplot as plt

import utils
import plane_seg
import comp

ti.init(arch=ti.gpu)

tivec3 = ti.types.vector(3, float)
mat3x3f = ti.types.matrix(3, 3, float)

img_rgb_path = "198.jpg"
img_rgb = skimage.io.imread(img_rgb_path)
img_rgb = img_rgb.astype(np.float32)
img_rgb_field = ti.field(dtype=ti.f32, shape=(img_rgb.shape[0], img_rgb.shape[1], 3))
img_rgb_field.from_numpy(img_rgb)
img_depth_gt_path = "198_depth_gt.png"
img_depth_gt = skimage.io.imread(img_depth_gt_path)
img_depth_gt = img_depth_gt.astype(np.float32)
dd = 1
img_depth_gt /= dd
img_depth_gt_field = ti.field(dtype=ti.f32, shape=(img_depth_gt.shape[0], img_depth_gt.shape[1]))
img_depth_gt_field.from_numpy(img_depth_gt)
img_objseg_gt_name = "198_obj_seg.png"
img_objseg_gt = skimage.io.imread(img_objseg_gt_name)
img_objseg_gt_field = ti.field(dtype=ti.f32, shape=(img_objseg_gt.shape[0], img_objseg_gt.shape[1]))
img_objseg_gt_field.from_numpy(img_objseg_gt)

field_point_cloud = ti.field(dtype=ti.f32, shape=(img_depth_gt.shape[0], img_depth_gt.shape[1], 3))

for i in range(100, 200):
    for j in range(300, 400):
        # print(f"i={i}, j={j}, img_depth_gt[i, j] = {img_depth_gt[i, j]}", end=' ')
        print(img_depth_gt[i, j], end=' ')
print()
print("img_depth_gt.shape[0] = ", img_depth_gt.shape[0], ", img_depth_gt.shape[1] = ", img_depth_gt.shape[1])
L = 10
invalid = -500

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

################################################################################################################

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

###########################################################################################################

# 得到seed patch 的数量,已完结
@ti.kernel
def get_seed_patch_num() -> int:
    ret = 0
    for i, j in img_depth_gt_field:
        if i+L >= img_depth_gt_field.shape[0]:
            continue
        if j+L >= img_depth_gt_field.shape[1]:
            continue
        if i % (2*L) != 0 or j % (2*L) != 0:
            continue
        canbe_seedpatch = True
        for k in range(L):
            for l in range(L):
                if ti.abs(img_depth_gt_field[i+k, j+l] - 0) < 1e-4:
                    canbe_seedpatch = False
                    continue
                if img_objseg_gt_field[i, j] != img_objseg_gt_field[i+k, j+l]:
                    canbe_seedpatch = False
                    continue
        if canbe_seedpatch:
            ret += 1
    return ret


seed_num = get_seed_patch_num()
print("seed_num = ", seed_num)
plane_seg_img2d = ti.field(dtype=ti.i32, shape=(img_rgb_field.shape[0], img_rgb_field.shape[1]))
plane_seg_img3d = ti.field(dtype=ti.i32, shape=(img_rgb_field.shape[0], img_rgb_field.shape[1], seed_num+2))
# 有seed_num个平面分割，在每一个平面idx下，有7个信息：idx, error, a, b, d, (x,y)
plane_list = ti.Vector.field(n=7, dtype=ti.f32, shape=(seed_num+2,1))

@ti.func
def seedidx_encode(H:int, W:int, a, b):
    max_size = ti.max(H, W)
    return a * max_size + b

tivecL = ti.types.vector(L, float)
tivecL2 = ti.types.vector(L*L, float)

# 得到初始的seed patch的分割mask
@ti.kernel
def get_seed_patch_seg():
    for i, j in img_depth_gt_field:
        if i+L >= img_depth_gt_field.shape[0]:
            continue
        if j+L >= img_depth_gt_field.shape[1]:
            continue
        if i % (2*L) != 0 or j % (2*L) != 0:
            continue
        canbe_seedpatch = True
        for k in range(L):
            for l in range(L):
                if ti.abs(img_depth_gt_field[i+k, j+l] - 0) < 1e-4:
                    canbe_seedpatch = False
                    continue
                if img_objseg_gt_field[i, j] != img_objseg_gt_field[i+k, j+l]:
                    canbe_seedpatch = False
                    continue
        if canbe_seedpatch:
            idx = seedidx_encode(img_depth_gt_field.shape[0], img_depth_gt_field.shape[1], i, j)
            for k in range(L):
                for l in range(L):
                    plane_seg_img2d[i+k, j+l] = idx
    return  

# 得到初始的seed_patch对应的plane_list
@ti.kernel
def get_seed_patch_seg2():
    for i, j in img_depth_gt_field:
        if i+L >= img_depth_gt_field.shape[0]:
            continue
        if j+L >= img_depth_gt_field.shape[1]:
            continue
        if i % (2*L) != 0 or j % (2*L) != 0:
            continue
        canbe_seedpatch = True
        for k in range(L):
            for l in range(L):
                if ti.abs(img_depth_gt_field[i+k, j+l] - 0) < 1e-4:
                    canbe_seedpatch = False
                    continue
                if img_objseg_gt_field[i, j] != img_objseg_gt_field[i+k, j+l]:
                    canbe_seedpatch = False
                    continue
        if canbe_seedpatch:
            curplane_idx = plane_seg_img2d[i, j]
            plane_list[curplane_idx, 0][0] = curplane_idx
            x_list = tivecL2([0]*(L*L))
            y_list = tivecL2([0]*(L*L))
            z_list = tivecL2([0]*(L*L))
            for k in range(L):
                for l in range(L):
                    x_list[k+l] = field_point_cloud[i+k, j+l, 0]
                    # print(f"x_list付了一个值: {field_point_cloud[i+k, j+l, 0]}")
                    y_list[k+l] = field_point_cloud[i+k, j+l, 1]
                    z_list[k+l] = field_point_cloud[i+k, j+l, 2]
            # for ii in range(L*L):
            #     print(x_list[ii], end=" ")
            X = utils.pdpd_least_square(L*L, x_list, y_list, z_list)
            # print(f"X = {X}")
            a = X[0]
            b = X[1]
            d = X[2]
            err = 0.0
            sum = 0.0
            for k in range(L):
                for l in range(L):
                    normal_n = tivec3([a,b,-1])
                    p_at0 = field_point_cloud[i+k, j+l, 0]
                    p_at1 = field_point_cloud[i+k, j+l, 1]
                    p_at2 = field_point_cloud[i+k, j+l, 2]
                    ndotp_plusd = normal_n[0] * p_at0 + normal_n[1] * p_at1 + normal_n[2] * p_at2 + d
                    sum = sum + ndotp_plusd * ndotp_plusd
            sum = sum / L / L
            if sum > 0 :
                sum = tm.sqrt(sum)
            err = sum
            # 有seed_num个平面分割，在每一个平面idx下，有7个信息：idx, error, a, b, d, (x,y)
            # plane_list = ti.Vector.field(n=7, dtype=ti.f32, shape=(seed_num,1))
            plane_list[curplane_idx, 0][0] = curplane_idx
            plane_list[curplane_idx, 0][1] = err
            plane_list[curplane_idx, 0][2] = a
            plane_list[curplane_idx, 0][3] = b
            plane_list[curplane_idx, 0][4] = d
            plane_list[curplane_idx, 0][5] = i
            plane_list[curplane_idx, 0][6] = j
            # print(f"curplane_idx = {curplane_idx}, a = {a}, b = {b}, d = {d}")
    return  
    
@ti.func
def cal_error(n:tivec3, p:tivec3, d:float) -> float:
    return ti.abs( n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d )

@ti.func
def cal_err_T(x:int, y:int, j,tao,lambdaa,H,W,alpha,k) -> float:
    d = img_depth_gt_field[x, y]
    t = ( tao * ( 1 - tm.exp(-(j/lambdaa)) ) ) 
    ret = t * t
    if j > H * W / k / k:
        ret = ret * alpha * d * d
    return ret


threshhold_recal_one = 100
threshhold_not_recal = 2500
@ti.kernel
def compute_seg() -> int:
    # plane_seg_img3d = ti.field(dtype=ti.i32, shape=(img_rgb_field.shape[0], img_rgb_field.shape[1], seed_num))
    # 有seed_num个平面分割，在每一个平面idx下，有7个信息：idx, error, a, b, d, (x,y)
    # plane_list = ti.Vector.field(n=7, dtype=ti.f32, shape=(seed_num,1))
    tao = 3
    lambdaa = 1
    alpha = 0.009
    kk = 20
    for i in range(2, seed_num+2):
        idx = int(plane_list[i, 0][0])
        x = int(plane_list[idx, 0][5])
        y = int(plane_list[idx, 0][6])
        top = 0
        down = 0
        right = 0
        left = 0
        if x - 1 >= 0:
            top = x - 1
        else:
            top = x
        if y - 1 >= 0:
            left = y - 1
        else :
            left = y
        if x + L < img_depth_gt_field.shape[0]:
            down = x + L
        else :
            down = x + L - 1
        if x + L < img_depth_gt_field.shape[1]:
            right = y + L
        else :
            right = y + L - 1
        growing = True 
        grow_stage_cnt = 1
        newtop = top
        newdown = down
        newleft = left
        newright = right

        while growing:
            last_grow_stage_cnt = grow_stage_cnt
            top = newtop
            down = newdown
            left = newleft
            right = newright
            # ****** ****** ****** code for recal ***** ****** ******
            Asum11 = 0.0
            Asum12 = 0.0
            Asum13 = 0.0
            Asum21 = 0.0
            Asum22 = 0.0
            Asum23 = 0.0
            Asum31 = 0.0
            Asum32 = 0.0
            Asum33 = 1.0
            Bsum1 = 0.0
            Bsum2 = 0.0
            Bsum3 = 0.0
            N = 0
            err_sum = 0.0

            a = plane_list[i, 0][2]
            b = plane_list[i, 0][3]
            d = plane_list[i, 0][4]
            err = plane_list[i, 0][1]

            # ***** ***** ****** code for recal end ***** ***** ***** 
            for j in range(top, down+1):
                for k in range(left, right+1):
                    a = plane_list[i, 0][2]
                    b = plane_list[i, 0][3]
                    d = plane_list[i, 0][4]
                    n = tivec3([a,b,-1])
                    if plane_seg_img3d[j, k, idx] == idx:
                        N += 1
                        xi = field_point_cloud[j, k, 0]
                        yi = field_point_cloud[j, k, 1]
                        zi = field_point_cloud[j, k, 2]
                        Asum11 += xi ** 2
                        Asum12 += xi * yi
                        Asum13 += xi
                        Asum21 += xi * yi
                        Asum22 += yi ** 2
                        Asum23 += yi
                        Asum31 += xi
                        Asum32 += yi
                        Bsum1 += xi * zi
                        Bsum2 += yi * zi
                        Bsum3 += zi
                        err_sum += ( a * xi + b * yi - zi + d ) ** 2
                        continue
                    cond1 = ( j-1 >= 0 and plane_seg_img3d[j-1, k, idx] == idx )
                    cond2 = ( j+1 <= plane_seg_img2d.shape[0] and plane_seg_img3d[j+1, k, idx] == idx )
                    cond3 = ( k-1 >= 0 and plane_seg_img3d[j, k-1, idx] == idx )
                    cond4 = ( k+1 <= plane_seg_img2d.shape[1] and plane_seg_img3d[j, k+1, idx] == idx )
                    
                    # print(f"position 1, x = {x}, y = {y}, top = {top}, down = {down}, left = {left}, right = {right}")
                    if not cond1 and not cond2 and not cond3 and not cond4:
                        continue
                    objcond1 = ( j-1 >= 0 and img_objseg_gt_field[j-1, k] == img_objseg_gt_field[j, k] )
                    objcond2 = ( j+1 <= plane_seg_img2d.shape[0] and img_objseg_gt_field[j+1, k] == img_objseg_gt_field[j, k] )
                    objcond3 = ( k-1 >= 0 and img_objseg_gt_field[j, k-1] == img_objseg_gt_field[j, k] )
                    objcond4 = ( k+1 <= plane_seg_img2d.shape[1] and img_objseg_gt_field[j, k+1] == img_objseg_gt_field[j, k] )
                    if not ( cond1 and objcond1 ) and not( cond2 and objcond2 ) and not( cond3 and objcond3 ) and not( cond4 and objcond4 ):
                        continue
                    
                    p_at0 = field_point_cloud[j, k, 0]
                    p_at1 = field_point_cloud[j, k, 1]
                    p_at2 = field_point_cloud[j, k, 2]
                    p = tivec3([p_at0, p_at1, p_at2])
                    p_err = cal_error(n,p,d)
                    err_T = cal_err_T(j, k, grow_stage_cnt,tao,lambdaa,img_depth_gt_field.shape[0],
                                      img_depth_gt_field.shape[1],alpha,kk)
                    # print(f"n = [{a}, {b}, -1], p = [{p_at0}, {p_at1}, {p_at2}], d = {d}, p_err = {p_err}, err_T = {err_T}")

                    # print("position 0")
                    if p_err > err_T:
                        continue

                    grow_stage_cnt += 1
                    # print("grow_stage_cnt = ",grow_stage_cnt)
                    plane_seg_img3d[j, k, idx] = idx
                    if j <= top and j > 0 : 
                        newtop = j - 1
                    if j >= down and j < img_depth_gt_field.shape[0] - 1:
                        newdown = j + 1
                    if k <= left and k > 0:
                        newleft = k - 1
                    if k >= right and k < img_depth_gt_field.shape[1] - 1:
                        newright = k + 1

                    N += 1
                    xi = field_point_cloud[j, k, 0]
                    yi = field_point_cloud[j, k, 1]
                    zi = field_point_cloud[j, k, 2]
                    Asum11 += xi ** 2
                    Asum12 += xi * yi
                    Asum13 += xi
                    Asum21 += xi * yi
                    Asum22 += yi ** 2
                    Asum23 += yi
                    Asum31 += xi
                    Asum32 += yi
                    Bsum1 += xi * zi
                    Bsum2 += yi * zi
                    Bsum3 += zi
                    err_sum += ( a * xi + b * yi - zi + d ) ** 2

                # end for k in range(left, right+1)
            # end for j in range(top, down+1)      
            if last_grow_stage_cnt == grow_stage_cnt:
                # if last_grow_stage_cnt == grow_stage_cnt:
                #     print(f"idx{i}: last_grow_stage_cnt == grow_stage_cnt")
                # else :
                #     print(f"idx{i}: grow_stage_cnt > 90000")
                growing = False
            
            if grow_stage_cnt < threshhold_not_recal:
                # 有seed_num个平面分割，在每一个平面idx下，有7个信息：idx, error, a, b, d, (x,y)
                # plane_list = ti.Vector.field(n=7, dtype=ti.f32, shape=(seed_num,1))
                A = mat3x3f([[Asum11, Asum12, Asum13], 
                    [Asum21, Asum22, Asum23], 
                    [Asum31, Asum32, N]])
                B = tivec3([Bsum1, Bsum2, Bsum3])
                A_ivs = A.inverse()
                X = A_ivs @ B
                err_sum /= N
                err_sum = tm.sqrt(err_sum)
                plane_list[idx, 0][1] = err_sum
                plane_list[idx, 0][2] = X[0]
                plane_list[idx, 0][3] = X[1]
                plane_list[idx, 0][4] = X[2]
                # print(f"old: [{a}, {b}, {d}, {err}], new: [{X[0]}, {X[1]}, {X[2]}, {err_sum}]")

        # end while growing
    return 3

evtual_seg_field = ti.field(dtype=int, shape=(plane_seg_img3d.shape[0], plane_seg_img3d.shape[1]))
@ti.kernel
def merge():
    for i, j in img_depth_gt_field:
        min_err = 2e9 + 0.0
        min_err_idx = -500
        for k in range(2, plane_seg_img3d.shape[2]):
            if plane_seg_img3d[i, j, k] != k:
                continue
            cur_plane_idx = k
            cur_err = plane_list[cur_plane_idx, 0][1]
            if min_err > cur_err:
                min_err = cur_err
                min_err_idx = k
        evtual_seg_field[i, j] = min_err_idx
    return 

@ti.kernel
def initfield():
    for i, j in plane_seg_img2d:
        plane_seg_img2d[i, j] = invalid
    return 

@ti.kernel
def initplane_seg_img3d():
    for i, j, k in plane_seg_img3d:
        plane_seg_img3d[i, j, k] = plane_seg_img2d[i, j]
    return 

def pyscope_idxmapping():
    np_planeseg_img2d = plane_seg_img2d.to_numpy()
    l = []
    for i in range(np_planeseg_img2d.shape[0]):
        for j in range(np_planeseg_img2d.shape[1]):
            if np_planeseg_img2d[i, j] not in l:
                l.append(np_planeseg_img2d[i, j])
    l = sorted(l)
    ret = {}
    idx = 1
    for val in l:
        ret[val] = idx
        idx += 1
        pass
    for i in range(np_planeseg_img2d.shape[0]):
        for j in range(np_planeseg_img2d.shape[1]):
            if np_planeseg_img2d[i, j] != invalid:
                np_planeseg_img2d[i, j] = ret[np_planeseg_img2d[i, j]]
    return np_planeseg_img2d

def print_plane_list():
    for i in range(seed_num):
        print(f"{i}: idx = {plane_list[i, 0][0]}, err = {plane_list[i, 0][1]}, a = {plane_list[i, 0][2]}\
              b = {plane_list[i, 0][3]}, d = {plane_list[i, 0][4]}, x = {plane_list[i, 0][5]}, y = {plane_list[i, 0][6]} ")
    return 

def pyscope_merge2():
    for i in range(img_depth_gt.shape[0]-1):
        for j in range(img_depth_gt.shape[1]-1):
            this_idx = np_evtual_seg_field[i, j]
            this_a = np_plane_list[this_idx, 0][2]
            this_b = np_plane_list[this_idx, 0][3]
            this_d = np_plane_list[this_idx, 0][4]
            right_idx = np_evtual_seg_field[i, j+1]
            right_a = np_plane_list[right_idx, 0][2]
            right_b = np_plane_list[right_idx, 0][3]
            right_d = np_plane_list[right_idx, 0][4]
            down_idx = np_evtual_seg_field[i+1, j]
            down_a = np_plane_list[down_idx, 0][2]
            down_b = np_plane_list[down_idx, 0][3]
            down_d = np_plane_list[down_idx, 0][4]
            if abs(this_a-right_a) < 1e-2 and abs(this_b-right_b) < 1e-2 and abs(this_d-right_d) < 1e-2:
                np_evtual_seg_field[i, j+1] = this_idx
            if abs(this_a-down_a) < 1e-2 and abs(this_b-down_b) < 1e-2 and abs(this_d-down_d) < 1e-2:
                np_evtual_seg_field[i+1, j] = this_idx
        # print(f"i = {i}")
    return
############################################################################################################

print("inv_K_T = ",inv_K_T)

# # 测试1：depth_to_space_point()测试通过！
# depth_to_space_point()
# pdpd_cloud = point_cloud_img.to_numpy()
# dpd_cloud = plane_seg.depth_to_space_point(img_depth_gt, inv_K_torch_T)
# dpd_cloud = dpd_cloud.numpy()
# if comp.comp_img(dpd_cloud, pdpd_cloud):
#     print("Depth to space point test: True")
# else :
#     print("Depth to space point test: false")

# 测试2：get seed patch, 编译已通过，逻辑上不太好测试，看效果是没问题的！
# initfield()
# get_seed_patch_seg()
# np_planeseg_img2d = pyscope_idxmapping()
# get_seed_patch_seg2()

# plane_seg_img2d.from_numpy(np_planeseg_img2d)
# seed_seg = plane_seg_img2d.to_numpy()
# img_save(seed_seg, "pdpd_seedseg.png", "pdpd_c_testseed.png")
# print("seed_seg img have saved as: pdpd_seedseg.png, pdpd_c_testseed.png" )


# 整体测试
# depth_to_space_point()
# initfield()
# get_seed_patch_seg()
# np_planeseg_img2d = pyscope_idxmapping()
# plane_seg_img2d.from_numpy(np_planeseg_img2d)
# get_seed_patch_seg2()
# plane_seg_img2d.from_numpy(np_planeseg_img2d)
# seed_seg = plane_seg_img2d.to_numpy()
# img_save(seed_seg, "pdpd_seedseg.png", "pdpd_c_testseed.png")
# print("seed_seg img have saved as: pdpd_seedseg.png, pdpd_c_testseed.png" )

# initplane_seg_img3d()

# start_time = time.time()
# a = compute_seg()
# end_time = time.time()
# print("compute_seg() have finished. time cost : ",end_time - start_time )
# merge()

# np_plane_list = plane_list.to_numpy()
# np_evtual_seg_field = evtual_seg_field.to_numpy()
# pyscope_merge2()
# evtual_seg_field.from_numpy(np_evtual_seg_field)
# np_evtual_seg = np.zeros((plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
# np_evtual_seg = evtual_seg_field.to_numpy()
# img_save(np_evtual_seg, "pdpd_evtual_seg.png", "pdpd_c_evtual_seg.png")
# print("evtual_seg img have saved as: pdpd_evtual_seg.png, pdpd_c_evtual_seg.png" )




print()


# plane_seg_10 = np.zeros((plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
# for i in range(plane_seg_img2d.shape[0]):
#     for j in range(plane_seg_img2d.shape[1]):
#         plane_seg_10[i, j] = plane_seg_img3d[i, j, 10]
# img_save(plane_seg_10, "pdpd_plane_seg_10.png", "pdpd_c_plane_seg_10.png")
# print("10 finished.")

# plane_seg_87 = np.zeros((plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
# for i in range(plane_seg_img2d.shape[0]):
#     for j in range(plane_seg_img2d.shape[1]):
#         plane_seg_87[i, j] = plane_seg_img3d[i, j, 87]
# img_save(plane_seg_87, "pdpd_plane_seg_87.png", "pdpd_c_plane_seg_87.png")
# print("87 finished.")

# plane_seg_109 = np.zeros((plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
# for i in range(plane_seg_img2d.shape[0]):
#     for j in range(plane_seg_img2d.shape[1]):
#         plane_seg_109[i, j] = plane_seg_img3d[i, j, 109]
# img_save(plane_seg_109, "pdpd_plane_seg_109.png", "pdpd_c_plane_seg_109.png")
# print("109 finished.")

# plane_seg_187 = np.zeros((plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
# for i in range(plane_seg_img2d.shape[0]):
#     for j in range(plane_seg_img2d.shape[1]):
#         plane_seg_187[i, j] = plane_seg_img3d[i, j, 187]
# img_save(plane_seg_187, "pdpd_plane_seg_187.png", "pdpd_c_plane_seg_187.png")
# print("187 finished.")

# plane_seg_233 = np.zeros((plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
# for i in range(plane_seg_img2d.shape[0]):
#     for j in range(plane_seg_img2d.shape[1]):
#         plane_seg_233[i, j] = plane_seg_img3d[i, j, 233]
# img_save(plane_seg_233, "pdpd_plane_seg_233.png", "pdpd_c_plane_seg_233.png")
# print("233 finished.")

# plane_seg_234 = np.zeros((plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
# for i in range(plane_seg_img2d.shape[0]):
#     for j in range(plane_seg_img2d.shape[1]):
#         plane_seg_234[i, j] = plane_seg_img3d[i, j, 234]
# img_save(plane_seg_234, "pdpd_plane_seg_234.png", "pdpd_c_plane_seg_234.png")
# print("234 finished.")

# plane_seg_256 = np.zeros((plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
# for i in range(plane_seg_img2d.shape[0]):
#     for j in range(plane_seg_img2d.shape[1]):
#         plane_seg_256[i, j] = plane_seg_img3d[i, j, 256]
# img_save(plane_seg_256, "pdpd_plane_seg_256.png", "pdpd_c_plane_seg_256.png")
# print("256 finished.")

# plane_seg_287 = np.zeros((plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
# for i in range(plane_seg_img2d.shape[0]):
#     for j in range(plane_seg_img2d.shape[1]):
#         plane_seg_287[i, j] = plane_seg_img3d[i, j, 287]
# img_save(plane_seg_287, "pdpd_plane_seg_287.png", "pdpd_c_plane_seg_287.png")
# print("287 finished.")

# plane_seg_352 = np.zeros((plane_seg_img2d.shape[0], plane_seg_img2d.shape[1]))
# for i in range(plane_seg_img2d.shape[0]):
#     for j in range(plane_seg_img2d.shape[1]):
#         plane_seg_352[i, j] = plane_seg_img3d[i, j, 352]
# img_save(plane_seg_352, "pdpd_plane_seg_352.png", "pdpd_c_plane_seg_352.png")
# print("352 finished.")

# cnt = 0
# for i in range(plane_seg_img2d.shape[0]):
#     for j in range(plane_seg_img2d.shape[1]):
#         # print(plane_seg_img3d[i, j, 352], end=' ')
#         cnt += 1
#         if cnt % 80 == 0:
#             # print()
#             continue
#         if plane_seg_img3d[i, j, 352] != 0:
#             print(plane_seg_img3d[i, j, 352])

print()

# print_seed_seg_np
# cnt = 0
# for i in range(seed_seg.shape[0]):
#     for j in range(seed_seg.shape[1]):
#         print(seed_seg[i, j], end=' ')
#         cnt += 1
#         if cnt % 80 == 0:
#             print()
#     if cnt > 2000 :
#         break

pass


