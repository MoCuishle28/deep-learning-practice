import torch

# 运算符号：
# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)	# 转换为 32bit 的 torch 张量数据
print('abs:' ,torch.abs(tensor))

# sin
print('sin:', torch.sin(tensor))

# mean
print('mean:', torch.mean(tensor))

print('矩阵运算')

# 矩阵运算:
# 矩阵相乘
data = [
		[1,2], 
		[3,4]
	]
tensor = torch.FloatTensor(data)
print('乘法:', torch.mm(tensor, tensor))