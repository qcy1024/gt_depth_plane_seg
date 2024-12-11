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
import skimage
from skimage import io,data
import taichi as ti

ti.init(arch=ti.cpu)

ti_struct_seed_list = ti.Struct.field({"error":float,"param":ti.math.vec3,"id":int,"pixels":ti.math.vec2,"pixel_nums":int},shape=3)
point_list = np.array([[[0,0],[0,1],[0,2]],[[0,0],[1,1],[0,2]],[[0,0],[2,1],[0,2]]])
ti_point_list = ti.Vector.field(n=2,dtype=int,shape=(3,3))
ti_point_list.from_numpy(point_list)
# print(ti_struct_seed_list[0])
ti_struct_seed_list[0].param = [1.1,2.2,2.23]
ti_struct_seed_list[1].param = [1.1,2.2,2.2334]
ti_struct_seed_list[2].param = [1.1,2.2,2.2343434]

ti_struct_seed_list[0].pixel_nums = 3
ti_struct_seed_list[1].pixel_nums = 3
ti_struct_seed_list[2].pixel_nums = 3



@ti.kernel
def test():
    for i in ti_struct_seed_list:
        print(i)
        print(ti_struct_seed_list[i].param)
        l = []
        for j in range(ti_struct_seed_list[i].pixel_nums):
            l.append(ti_point_list[i,j])
        print(l)
    return 

test()
