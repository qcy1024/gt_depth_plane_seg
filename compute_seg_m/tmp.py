import torch
import queue

# 假设seed_list是你的字典
seed_list = {
    1: [3.14, [1.0, 2.0, 3.0], 5, True],
    2: [2.5, [0.0, 0.0, 0.0], 10, False],
    3: [1.5, [4.0, 5.0, 6.0], 15, True],
    4: [1.2, [1,1,1], 2, False ],
    5: [2.2, [1,1,1], 2, False],
    6: [0,8, [1,1,1], 2, True]
}

# 选取第四个值为False且第一个float类型值最小的元素
selected_value = min(
    [value for value in seed_list.values() if isinstance(value[0], float) and value[3] == False],
    key=lambda x: x[0]
)

# print(selected_value)   # [1.2, [1, 1, 1], 2, False]

# print([1,2]==[1,2])
# print([1,2]==[1,3])

