import numpy as np

def least_sqaure(N,x_list,y_list,z_list):
    A = np.array([[sum(x_list ** 2), sum(x_list * y_list), sum(x_list)],
                [sum(x_list * y_list), sum(y_list ** 2), sum(y_list)],
                [sum(x_list), sum(y_list), N]])

    B = np.array([[sum(x_list * z_list), sum(y_list * z_list), sum(z_list)]])
    X = np.linalg.solve(A, B.T)
    return X

