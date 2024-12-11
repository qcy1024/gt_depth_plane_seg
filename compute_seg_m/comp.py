import numpy as np
import torch

from PIL import Image

import skimage
from skimage import io,data

import taichi as ti
import taichi.math as tm

# ti.init(arch=ti.gpu)

def comp_img(lhs, rhs):
    if lhs.shape != rhs.shape :
        print("shape not same")
        return False
    for i in range(lhs.shape[0]):
        for j in range(lhs.shape[1]):
            sum = 0.0
            for k in range(3):
                sum += abs(lhs[i,j][k] - rhs[i,j][k])
                # print("0: ", abs(lhs[i,j][0] - rhs[i,j][0]))
                # print("1: ", abs(lhs[i,j][1] - rhs[i,j][1]))
                # print("2: ", abs(lhs[i,j][2] - rhs[i,j][2]))
            if sum > 1e-4 :
                print("sum=", sum, "i=", i, "j=", j)
                return False
    return True


if __name__ == "__main__":
    


    pass

