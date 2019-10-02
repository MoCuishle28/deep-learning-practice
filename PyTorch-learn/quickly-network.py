import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 100*2 的 全为 1 的张量 (基数?)
n_data = torch.ones(100, 2)

# class0 特征值 (2维?) (normal 是在某个数附近随机生成?)
x0 = torch.normal(2*n_data, 1)		# class0 x data (tensor), shape = (100, 2)
# 标签 (100 个 0)
y0 = torch.zeros(100)				# class0 y data (tensor), shape = (100, 1)

# class1 特征值
x1 = torch.normal(-2*n_data, 1)		# class1 x data (tensor), shape = (100, 1)
# 标签 (100 个 1)
y1 = torch.ones(100)				# class1 y data (tensor), shape = (100, 1)
	
# 将 x0, x1 合并在一起 方便训练
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)	# FloatTensor = 32 bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)	# LongTensor = 64 bit integer

x, y = Variable(x), Variable(y)


# 快速搭建神经网络 (效果和自定义class一样)
# 只要在这里面垒神经层就好
net = torch.nn.Sequential(
	# 设置 3 层 (激励函数也看作一层)
	torch.nn.Linear(2, 10),		# 2个输入 10个神经元(输出)
	torch.nn.ReLU(),
	torch.nn.Linear(10, 2),
)

# 输出 [1, 0] --> class0 (第0个位置为1 则是class0)
# 输出 [0, 1] --> class1
print(net)

plt.ion()	# 实时画图
plt.show()

# 优化神经网络参数 传入参数： 神经网络所有参数, lr-->学习效率(一般 < 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)

# 计算误差的手段(损失函数) 这里用的是 CrossEntropyLoss (多用于分类问题, 计算得到的是概率)
loss_func = torch.nn.CrossEntropyLoss()

# 训练100次
for t in range(50):
	out = net(x)		# out 是每一步得到的分类值 (在经过 softmax 之前还不是概率)

	loss = loss_func(out, y)	# 计算与真实值 y 之间的误差

	# 优化步骤
	optimizer.zero_grad()	# 先将梯度降为0
	loss.backward()			# 反向传递
	optimizer.step()		# 再用 optimizer 以0.5学习效率 优化梯度

	if t % 2 == 0:
		plt.cla()
		# 后面的 [1] 返回的是最大值的位置, [0] 则会返回最大值
		prediction = torch.max(F.softmax(out, dim=1), 1)[1] # dim=1 -> 对每一列 softmax
		pred_y = prediction.data.numpy().squeeze()
		target_y = y.data.numpy()

		plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0)
		accuracy = sum(pred_y == target_y) / 200
		plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size':20, 'color':'red'})
		plt.pause(0.05)


plt.ioff()
plt.show()