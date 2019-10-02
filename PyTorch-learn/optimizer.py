import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 超参数
LR = 0.01
BATCH_SIZE = 32		# 20 效果更好
EPOCH = 12			# 80

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# plt.scatter(x.numpy(), y.numpy(), lw=1)
# plt.show()

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

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


# 建立4个神经网络分别用不同的优化器优化
net_SGD = Net(1, 20, 1)
net_Momentum = Net(1, 20, 1)
net_RMSprop = Net(1, 20, 1)
net_Adam = Net(1, 20, 1)

nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# 建立4个优化器
opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)	# 加上了 momentum 参数
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_his = [[], [], [], []]	# 记录误差

for epoch in range(EPOCH):
	print('Epoch: ', epoch)
	for step, (b_x, b_y) in enumerate(loader):          # for each training step
		for net, opt, l_his in zip(nets, optimizers, losses_his):
			output = net(b_x)              		# get output for every net
			loss = loss_func(output, b_y)  		# compute loss for every net
			opt.zero_grad()                		# clear gradients for next train
			loss.backward()                		# backpropagation, compute gradients
			opt.step()                     		# apply gradients
			l_his.append(loss.data.numpy())     # loss recoder


labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()