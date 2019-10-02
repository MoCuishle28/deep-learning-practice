import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2], [3,4]])
variable = Variable(tensor, requires_grad = True)	# requires_grad = True 则设置为反向传播计算的节点

print(tensor)
print(variable)

t_out = torch.mean(tensor * tensor)		# X^2
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

v_out.backward()		# 误差反向传递
print(variable.grad)	# 梯度