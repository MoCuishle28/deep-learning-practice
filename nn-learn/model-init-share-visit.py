import torch
from torch import nn
from torch.nn import init


net = nn.Sequential(
	nn.Linear(4, 3), 
	nn.ReLU(), 
	nn.Linear(3, 1))  # pytorch已进行默认初始化


# X = torch.rand(2, 4)
# Y = net(X).sum()
# print(Y)

# 访问模型参数
print(type(net.named_parameters()))
for name, param in net.named_parameters():
	print(name, param.size())

# 访问第 0 层的参数
print('---')
for name, param in net[0].named_parameters():
	print(name, param.size(), type(param))

print("---init---")
# 初始化模型参数
for name, param in net.named_parameters():
	if 'weight' in name:
		init.normal_(param, mean=0, std=0.01)
		print(name, param.data)
