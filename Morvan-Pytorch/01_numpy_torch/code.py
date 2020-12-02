"""
numpy转torch tensor
torch tensor转numpy

"""
import torch
import numpy as np

# transfer
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    'numpy', np_data,
    '\ntorch', torch_data,
    '\ntensor2array', tensor2array
)

# abs、sin、mean
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32bit

print(
    '\nabs',
    '\nnumpy', np.abs(data),
    '\ntorch', torch.abs(tensor)
)

# matrix calculate
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
data = np.array(data)
print(
    'matmul',
    '\nnumpy', np.matmul(data, data),
    '\ntorch', torch.mm(tensor, tensor)
)

print(
    'dot',
    '\nnumpy', data.dot(data),
    # '\ntorch', tensor.dot(tensor)  # 只支持一维，现在版本弃用
)

