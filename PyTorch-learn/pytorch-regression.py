import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 将1维的向量变为2维的数据 ([1,2,3] --> [[1,2,3]]) 因为 torch 只处理二维数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)	# x data (tensor), shape=(100,1)
y = x.pow(2) + 0.2*torch.rand(x.size())		# noisy y data (tensor), shape=(100,1)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


# 神经网络
class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		"""
		n_features: 多少个输入
		n_hidden:	多少个隐藏层神经元
		n_output:	多少个输出
		"""
		super(Net, self).__init__()
		# 输入个数, 输出个数
		self.hidden = torch.nn.Linear(n_feature, n_hidden)	# 隐藏层神经网络
		self.predict = torch.nn.Linear(n_hidden, n_output)	# 预测的神经层


	def forward(self, x):
		"""前向传递过程"""
		# 数据 x 经过 hidden_layer 再经过激励函数
		x = torch.relu(self.hidden(x))
		# 再经过输出层 后输出 (预测问题的输出一般不用激励函数, 不然预测函数图像会有截断)
		x = self.predict(x)
		return x


net = Net(1, 10, 1)
print(net)

plt.ion()	# 实时画图
plt.show()

# 优化神经网络参数 传入参数： 神经网络所有参数, lr-->学习效率(一般 < 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# 计算误差的手段(损失函数) 这里用的是均方差
loss_func = torch.nn.MSELoss()

# 训练100次
for t in range(100):
	prediction = net(x)		# prediction 是每一步得到的预测值

	loss = loss_func(prediction, y)	# 计算与真实值 y 之间的误差

	# 优化步骤
	optimizer.zero_grad()	# 先将梯度降为0
	loss.backward()			# 反向传递
	optimizer.step()		# 再用 optimizer 以0.5学习效率 优化梯度

	loss_num = loss.data.numpy()
	if t % 5 == 0:
		plt.cla()
		plt.scatter(x.data.numpy(), y.data.numpy())
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
		plt.text(0.5, 0, 'Loss=%.4f' % loss_num.tolist(), fontdict={'size':15, 'color':'red'})
		plt.pause(0.1)


plt.ioff()
plt.show()