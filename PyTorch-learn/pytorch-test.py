import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))	# 2行3列 的数据

torch_data = torch.from_numpy(np_data)

print(np_data)
print('------')
print(torch_data)
print('------')
print(torch_data.numpy())