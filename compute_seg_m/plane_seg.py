# 保存图片： skimage.io.imsave("output.png",img)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision import transforms
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import math
import torch
# from skimage.segmentation import all_felzenszwalb as felz_seg
from PIL import Image

import skimage
from skimage import io,data

from utils import least_sqaure
import queue
import taichi as ti

# 数据集中，图片的shape是(H,W,3)，depth_gt的shape是(H,W)，normal_gt的shape是(H,W,3)

def plane_seg_map_to_255(img_seg):
    dic = {}
    idx = 255
    # print("img_seg.shape=",img_seg.shape)
    for i in range(img_seg.shape[0]):
        for j in range(img_seg.shape[1]):
            # print("i=",i,", j=",j,", img_seg[i][j]=",img_seg[i][j])
            # print("dic = ", dic)
            if img_seg[i][j].item() in dic:
                continue
            else :
                dic[img_seg[i][j]] = idx
                idx -= 1
            # print("dic = ", dic)
    for i in range(img_seg.shape[0]):
        for j in range(img_seg.shape[1]):
            img_seg[i][j] = dic[img_seg[i][j].item()]
    print("map done. Maped img_seg = ",img_seg)
    return img_seg

# img_seg = np.array([[12,23,31],[41,54,65]])
# plane_seg_map_to_255(img_seg)

# 找到img_depth_gt中有多少个L*L的区域，这L^2个像素的深度标注都不为0，返回这个数量
def compute_vailable_seed_cnt(img_depth_gt,L):
    cnt = 0
    for i in range(0,img_depth_gt.shape[0]-L,L):
        for j in range(0,img_depth_gt.shape[1]-L,L):
            have_hole = False
            for k in range(i,i+L):
                for l in range(j,j+L) :
                    # print(img_depth_gt[k][l],end=' ')
                    if img_depth_gt[k][l] == 0:
                        have_hole = True
                # print()
            if not have_hole:
                cnt += 1
    return cnt

def complete_dpeth_gt(img_depth_gt):
    img_depth_gt_ret = img_depth_gt
    for i in range(img_depth_gt.shape[0]-3):
        for j in range(img_depth_gt.shape[1]-3):
            cnt = 0
            if img_depth_gt[i+1][j+1] != 0:
                continue
            sum = 0
            for k in range(3):
                for l in range(3):
                    if k == 1 and l == 1:
                        continue
                    if img_depth_gt[i+k][j+l] == 0:
                        cnt += 1
                    sum += img_depth_gt[i+k][j+l]
            if cnt <= 4:
                img_depth_gt_ret[i+1][j+1] = sum / (8-cnt)
    return img_depth_gt_ret

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

# 输入深度gt图，初始seed patch的大小L，图中对应的空间点坐标
# 返回初始的seed patch、seed patch的数量以及所有seed patch的list
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

# tensor, tensor, float
def cal_error(n_normal,p,d):
    ret = torch.dot(n_normal,p) + d
    ret = ret.item()
    ret = abs(ret)
    return ret

# val, val, val, [[]]
def cal_error_all(a,b,d,P,p_num,point_cloud):

    N = p_num
    x_list = []
    y_list = []
    z_list = []
    n_normal = torch.tensor([a,b,-1])
    sum = 0.0
    
    for i in range(p_num):
        u = P[i][0]
        v = P[i][1]
        pi = point_cloud[u][v]
        x_list.append(pi[0])
        y_list.append(pi[1])
        z_list.append(pi[2])
        n_dot_p_plus_d = torch.dot(pi,n_normal) + d
        sum = sum + n_dot_p_plus_d * n_dot_p_plus_d
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)
    X = least_sqaure(N,x_list,y_list,z_list)
    sum /= p_num
    sum = math.sqrt(sum)
    err = sum
    return X[0].item(), X[1].item(), X[2].item(), err

def cal_err_T(d,j,tao,lambdaa,H,W,alpha,k):
    d = d.item()
    t = ( tao * ( 1 - math.exp(-(j/lambdaa)) ) ) 
    ret = t * t
    if j > H * W / k / k:
        ret = ret * alpha * d * d
    return ret
        

def get_seg_plane(img_depth_gt,L,inv_K_T,img_obj_seg_gt):
    
    tao = 1
    lambdaa = 1
    alpha = 0.009
    k = 20
    
    point_cloud = depth_to_space_point(img_depth_gt,inv_K_T)
    print("point cloud got...")
    # plane_list.val: [err,[a,b,d],seg_idx,growed,points,point_nums]
    init_plane_seg, init_seg_cnt, plane_list = get_seed_patch_seg(img_depth_gt,L,point_cloud,img_obj_seg_gt)
    print("plane list got...")
    init_plane_seg = init_plane_seg.numpy()
    seed_seg_png_file_name = "seed_seg_nyu2_198.png"
    c_seed_seg_png_file_name = "c_seed_seg_nyu2_198.png"
    img_save(init_plane_seg,seed_seg_png_file_name,c_seed_seg_png_file_name)
    
    print("seed_seg got. png save as: ",seed_seg_png_file_name)
    print("initial seed patch cnt: ",init_seg_cnt)
    init_plane_seg = init_plane_seg.tolist()
    plane_seg = init_plane_seg
    # print("plane_seg=init_plane_seg,type:",type(plane_seg))   # list
    seg_cnt = init_seg_cnt
    err_T = 1
    for i in range(init_seg_cnt):
        # [err,[a,b,d],seg_idx,growed,points,point_nums]
        selected_plane_info = min(
        [value for value in plane_list.values() if isinstance(value[0], float) and value[3] == False],
        key=lambda x: x[0]
        )
        # print("selected_plane_info = ",selected_plane_info)
        cur_plane_idx = selected_plane_info[2]
        if selected_plane_info[5] == 0:
            plane_list[cur_plane_idx][3] = True
            seg_cnt -= 1
            continue
        
        visited = [[False] * len(plane_seg[0]) for _ in range(len(plane_seg))]
        # print("visited = ",visited)
        q = queue.Queue()
        src_point = selected_plane_info[4][0]
        q.put(src_point)
        # print(q.get())
        
        # for j in range(selected_plane_info[5]):
        #     q.put(selected_plane_info[4][j])
        #     visited[selected_plane_info[4][j][0]][selected_plane_info[4][j][1]] = True
        
        print(i+1,"th plane growing, plane idx = ",selected_plane_info[2],"src_point = ",selected_plane_info[4][0])
        grow_stage_cnt = 1
        # err_T = 1
        while not q.empty():
            pos = q.get()
            # cur_seg_idx = plane_seg[pos[0]][pos[1]]
            re_cal = False
            for ix in range(-1,2):
                for iy in range(-1,2):
                    if ix + pos[0] < 0 or iy + pos[1] < 0 or ix + pos[0] >= img_depth_gt.shape[0] or iy + pos[1] >= img_depth_gt.shape[1]:
                        continue
                    if ix == iy == 0 :
                    # if ix == iy == 0 or ix == iy == -1 or ix == iy == 1:
                        continue
                    if visited[pos[0]+ix][pos[1]+iy]:
                        continue
                    if img_depth_gt[pos[0]+ix][pos[1]+iy] == 0:
                        continue
                    if plane_seg[pos[0]+ix][pos[1]+iy] != 0 and plane_list[plane_seg[pos[0]+ix][pos[1]+iy]][3] == True:
                        continue
                    if img_obj_seg_gt[pos[0]][pos[1]] != img_obj_seg_gt[pos[0]+ix][pos[1]+iy]:
                        continue
                    
                    a = plane_list[cur_plane_idx][1][0]
                    b = plane_list[cur_plane_idx][1][1]
                    d = plane_list[cur_plane_idx][1][2]
                    n_normal = torch.tensor([a,b,-1])
                    p = point_cloud[pos[0]+ix,pos[1]+iy]
                    # tensor, tensor, float
                    # print("type(n_normal)=",type(n_normal),"type(p)=",type(p),"type(d)=",type(d))
                    p_err = cal_error(n_normal,p,d)
                    
                    err_T = cal_err_T(img_depth_gt[pos[0]+ix][pos[1]+iy],grow_stage_cnt,tao,lambdaa,img_depth_gt.shape[0],
                                      img_depth_gt.shape[1],alpha,k)
                   
                    # if grow_stage_cnt % 10 == 0:
                    #     print("image_depth_gt[pos[0]+ix][pos[1]+iy] = ",img_depth_gt[pos[0]+ix][pos[1]+iy],"type=",type(img_depth_gt[pos[0]+ix][pos[1]+iy]))
                    
                    # print("grow_stage_cnt = ",grow_stage_cnt,"err_T = ",err_T,"p_err = ",p_err)    
                    if p_err > err_T:
                        # print("grow_stage_cnt = ",grow_stage_cnt,"err_T = ",err_T,"p_err = ",p_err)   
                        continue
                    
                    re_cal = True
                    grow_stage_cnt += 1
                    nxt_step_plane_idx = plane_seg[pos[0]+ix][pos[1]+iy]
                    if nxt_step_plane_idx != 0:
                        plane_list[nxt_step_plane_idx][4].remove([pos[0]+ix,pos[1]+iy])
                        plane_list[nxt_step_plane_idx][5] -= 1
                        # if plane_list[nxt_step_plane_idx][5] == 0:
                            # print("现在有一个种子像素减少为0了，它的plane idx是: ",nxt_step_plane_idx)
                    q.put([pos[0]+ix,pos[1]+iy])
                    plane_list[cur_plane_idx][4].append([pos[0]+ix,pos[1]+iy])
                    plane_list[cur_plane_idx][5] += 1
                    plane_seg[pos[0]+ix][pos[1]+iy] = cur_plane_idx
                    
                    visited[pos[0]+ix][pos[1]+iy] = True
                    if grow_stage_cnt <= 100:
                        a = selected_plane_info[1][0]
                        b = selected_plane_info[1][1]
                        d = selected_plane_info[1][2]
                        new_a, new_b, new_d, new_err = cal_error_all(a,b,d,selected_plane_info[4],selected_plane_info[5],point_cloud)
                        plane_list[cur_plane_idx][0] = new_err
                        plane_list[cur_plane_idx][1][0] = new_a
                        plane_list[cur_plane_idx][1][1] = new_b
                        plane_list[cur_plane_idx][1][2] = new_d
            # end for ix in range(-1,2)
            # [err,[a,b,d],seg_idx,growed,points,point_nums]
            
            if re_cal and selected_plane_info[5] < 1800 and grow_stage_cnt > 100:
                a = selected_plane_info[1][0]
                b = selected_plane_info[1][1]
                d = selected_plane_info[1][2]
                new_a, new_b, new_d, new_err = cal_error_all(a,b,d,selected_plane_info[4],selected_plane_info[5],point_cloud)
                plane_list[cur_plane_idx][0] = new_err
                plane_list[cur_plane_idx][1][0] = new_a
                plane_list[cur_plane_idx][1][1] = new_b
                plane_list[cur_plane_idx][1][2] = new_d
                
                # print("num_points = ",selected_plane_info[5],"new err = ",new_err)
        # end while not q.empty():
        plane_list[cur_plane_idx][3] = True
        # print("执行了 plane_list[{}][3] = True".format(selected_plane_info[2]))
        
        # 保存每一个平面生长完之后的图片
        # seg_png_file_name = "seg_nyu2_25_{}th_plane_growned_L_{}_T_{}.png".format(i+1,L,tao)
        # c_seg_png_file_name = "c_seg_nyu2_25.png"
        # plane_seg = np.array(plane_seg)
        # if plane_list[cur_plane_idx][5] > 300:
        #     # print(type(plane_seg))
        #     img_save(plane_seg,seg_png_file_name,c_seg_png_file_name)
            # print("current plane segment saved as ",seg_png_file_name)
        
        
    # end for i in range(init_seg_cnt):
    
    # result = "plane_seg_22_L_{}_T_{}".format(L, T)
    np_plane_seg = np.array(plane_seg)
    np_plane_seg = plane_seg_map_to_255(np_plane_seg)
    print("np_plane_seg.type=",type(np_plane_seg))
    plane_seg_png_file_name = "plane_seg_xie_src_nyu2_198_L_{}_tao_{}_dd_{}_T_{}.png".format(L,tao,dd,err_T)
    c_plane_seg_png_file_name = "c_plane_seg_xie_src_nyu2_198_L_{}_tao_{}_dd_{}_T_{}.png".format(L,tao,dd,err_T)
    img_save(np_plane_seg,plane_seg_png_file_name,c_plane_seg_png_file_name)
    print("plane seg got. png save as: ",plane_seg_png_file_name)
    
    # plane_seg_np = plane_seg.cpu().numpy()
    
    # 保存原始的图像数组
    # original_np_plane_seg_file_name = 'original_np_plane_seg_22_L_{}_T_{}.npy'.format(L,err_T)
    # np.save(original_np_plane_seg_file_name,np_plane_seg)
    # print("original np plane seg save as:",original_np_plane_seg_file_name)

# end def 

img_depth_gt_path = "198_depth_gt.png"
img_depth_gt = skimage.io.imread(img_depth_gt_path)
# print("img_depth_gt.shape=",img_depth_gt.shape)     # (375,1242) img.shape: (375,1242,3)
img_depth_gt = img_depth_gt.astype(np.float32)
img_depth_gt = torch.from_numpy(img_depth_gt)
img_obj_seg_gt_name = "198_obj_seg.png"
img_obj_seg_gt = skimage.io.imread(img_obj_seg_gt_name)

dd = 1
img_depth_gt /= dd
L = 10
# 9.842439e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.808141e+02 2.331966e+02 0.000000e+00 0.000000e+00 1.000000e+00
# kitti
# inv_K = torch.tensor( [ [9.842439e+02,0.000000e+00,6.900000e+02],
#                         [0.000000e+00,9.808141e+02,2.331966e+02],
#                         [0.000000e+00,0.000000e+00,1.000000e+00] ])

inv_K = torch.tensor( [ [5.8262448167737955e+02,0.000000e+00,3.1304475870804731e+02],
                        [0.000000e+00,5.8269103270988637e+02,2.3844389626620386e+02],
                        [0.000000e+00,0.000000e+00,1.000000e+00] ])
inv_K_T = torch.inverse(inv_K)
print("inv_K_T = ",inv_K_T)
# point_cloud: (H,W,3)

# complete depth_gt
# print("completing depthgt...")
# img_depth_gt = complete_dpeth_gt(img_depth_gt)
# img_depth_gt = img_depth_gt.numpy()
# img_save(img_depth_gt,"completedScene1.png","c_completedScene1.png")
# print("depth_gt_completed, shape=",img_depth_gt.shape)


# err_T = 5
get_seg_plane(img_depth_gt,L,inv_K_T,img_obj_seg_gt)


                    

















# aligned_norm 是 normal_est_norm(预测出的经单位化后的法线)， D是distance_est[:, 0]
def compute_seg(rgb, aligned_norm, D):
        """
        inputs:
            rgb                 b, 3, H, W
            aligned_norm        b, 3, H, W
            D                   b, H, W

        outputs:
            segment                b, 1, H, W
            planar mask        b, 1, H, W
        """
        # rgb.shape= torch.Size([1, 3, 352, 1120]) 
        # aligned_norm.shape= torch.Size([1, 3, 352, 1120]) D.shape= torch.Size([1, 352, 1120])
        # print("compute_seg中，rgb.shape=",rgb.shape,"aligned_norm.shape=",aligned_norm.shape,"D.shape=",D.shape)
        b, _, h, w  = rgb.shape     # b:1, h:352, w:1120
        device = rgb.device

        # compute cost
        pdist = nn.PairwiseDistance(p=2)
        # pdist函数计算两个张量的欧氏距离
        rgb_down = pdist(rgb[:, :, 1:], rgb[:, :, :-1])         # torch.Size([1, 3, 351])
        rgb_right = pdist(rgb[:, :, :, 1:], rgb[:, :, :, :-1]) # rgb_right.shape= torch.Size([1, 3, 352])
        
        #print("troch.stack()前，rgb_down.shape=",rgb_down.shape,"rgb_right.shape=",rgb_right.shape)
        
        # def normalize(a):
        #   return (a - a.min())/(a.max() - a.min() + 1e-8)
        rgb_down = torch.stack([normalize(rgb_down[i]) for i in range(b)])
        rgb_right = torch.stack([normalize(rgb_right[i]) for i in range(b)])

        #print("troch.stack()后，rgb_down.shape=",rgb_down.shape,"rgb_right.shape=",rgb_right.shape)

        D_down = abs(D[:, 1:] - D[:, :-1])
        D_right = abs(D[:, :, 1:] - D[:, :, :-1])
        
        norm_down = pdist(aligned_norm[:, :, 1:], aligned_norm[:, :, :-1])
        norm_right = pdist(aligned_norm[:, :, :, 1:], aligned_norm[:, :, :, :-1])
        
        #print("torch.stack后D_down.shape=",D_down.shape,"D_right.shape=",D_down.shape,D_right.shape)
        #print("torch.stack前,norm_down.shape=",D_down.shape,"norm_right.shape=",norm_down.shape,norm_right.shape)

        D_down = torch.stack([normalize(D_down[i]) for i in range(b)])
        norm_down = pdist(aligned_norm[:, :, 1:].permute(0, 2, 3, 1), aligned_norm[:, :, :-1].permute(0, 2, 3, 1))


        D_right = torch.stack([normalize(D_right[i]) for i in range(b)])
        norm_right = pdist(aligned_norm[:, :, :, 1:].permute(0, 2, 3, 1), aligned_norm[:, :, :, :-1].permute(0, 2, 3, 1))
        
        #print("torch.stack后,D_down.shape=",D_down.shape,"D_right.shape=",D_down.shape,D_right.shape)
        #print("torch.stack后,norm_down.shape=",D_down.shape,"norm_right.shape=",norm_down.shape,norm_right.shape)

        #print("utils.py515行中，D_down.shape=",D_down.shape,"norm_down.shape=",norm_down.shape)
        
        normD_down = D_down + norm_down
        normD_right = D_right + norm_right

        normD_down = torch.stack([normalize(normD_down[i]) for i in range(b)])
        normD_right = torch.stack([normalize(normD_right[i]) for i in range(b)])

        # get max from (rgb, normD)
        # cost_down = torch.stack([rgb_down, normD_down])
        # cost_right = torch.stack([rgb_right, normD_right])
        # cost_down, _ = torch.max(cost_down, 0)
        # cost_right, _ = torch.max(cost_right, 0)
        cost_down = normD_down
        cost_right = normD_right
        # cost_down = rgb_down
        # cost_right = rgb_right

        # get dissimilarity map visualization
        dst = cost_down[:,  :,  : -1] + cost_right[ :, :-1, :]
        
        # felz_seg
        cost_down_np = cost_down.detach().cpu().numpy()
        cost_right_np = cost_right.detach().cpu().numpy()
        # from skimage.segmentation import all_felzenszwalb as felz_seg
        # felz_seg的参数：all_felzenszwalb(down_cost, right_cost, dright_cost, uright_cost, height, width, scale=1, sigma=0.8, min_size=20)
        # 返回的segment是一个与输入图像相同大小的整数张量，用不同的整数值代表的不同的分割区域。如：0，1，2等等。
        segment = torch.stack([torch.from_numpy(felz_seg(normalize(cost_down_np[i]), normalize(cost_right_np[i]), 0, 0, h, w, scale=2, min_size=200)).to(device) for i in range(b)])
        segment += 1
        # 这行代码使用了 unsqueeze() 函数，将 segment 张量在第一个维度（索引为1）上插入了一个新维度。假设 segment 张量的形状是 (height, width)，
        # 经过 unsqueeze(1) 操作后，segment 张量的形状变为 (height, 1, width)
        segment = segment.unsqueeze(1)
        
        # generate mask for segment with area larger than 200
        
        # max_num为segment分割中标签的数量。分割从0开始，0-9，就是有10个分割
        max_num = segment.max().item() + 1

        area = torch.zeros((b, max_num)).to(device)
        area.scatter_add_(1, segment.view(b, -1), torch.ones((b, 1, h, w)).view(b, -1).to(device))

        # 这段代码首先定义了一个阈值 planar_area_thresh，然后根据这个阈值生成了一个二值掩码 valid_mask，用于表示哪些分割区域的像素数量超过了阈值。
        # 接着，代码使用 torch.gather() 函数根据 segment 张量中的分割标签，从 valid_mask 中提取对应的二值掩码，并将结果保存在 planar_mask 张量
        # 中。最后，将 planar_mask 张量的形状重新调整为 (b, 1, h, w)，以便与其他张量具有相同的形状。
        # 这样，planar_mask 张量中的每个元素表示对应位置的像素是否属于具有足够大面积的平面区域。
        planar_area_thresh = 200
        valid_mask = (area > planar_area_thresh).float()
        planar_mask = torch.gather(valid_mask, 1, segment.view(b, -1))
        planar_mask = planar_mask.view(b, 1, h, w)

        planar_mask[:, :, :8, :] = 0
        planar_mask[:, :, -8:, :] = 0
        planar_mask[:, :, :, :8] = 0
        planar_mask[:, :, :, -8:] = 0

        # 每一个像素属于的分割区域(标签)、每一个像素是否属于具有足够大面积的平面区域掩码、
        return segment, planar_mask, dst.unsqueeze(1)
    
def normalize(a):
    return (a - a.min())/(a.max() - a.min() + 1e-8)


# rgb = torch.rand(1,3,352,1120)
# aligned_norm = torch.rand(1,3,352,1120)
# D = torch.rand(1,352,1120)
# print("rgb.shape = ",rgb.shape)                             #([1,1,352,1120])
# print("aligned_norm.shape=",aligned_norm.shape)             #([1,1,352,1120])
# print("D.shape=",D.shape)                                   #([1,1,352,1120])
# segment, planar_mask, dissimilarity_map = compute_seg(rgb, aligned_norm, D)

# print("segment.shape=",segment.shape)                       #([1,1,352,1120])
# print("planar_mask.shape=",planar_mask.shape)               #([1,1,352,1120])
# print("dissimilarity_map.shape=",dissimilarity_map.shape)   #([1,1,351,1119])


