import torch.nn.functional as F
import torch
import numpy as np


class Net(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.input_layer = torch.nn.Linear(3, 8)
		self.output_layer = torch.nn.Linear(8, 3)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
		self.LossFunc = torch.nn.CrossEntropyLoss()
		# self.LossFunc = torch.nn.MSELoss()

	def train(self, data, labels):
		for _ in range(3):
			y = self(data)	# 输出的是 (8,3)
			loss = self.LossFunc(y, labels)
			print('loss:', loss)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()


	def forward(self, X):
		hidden = F.relu(self.input_layer(X))
		# 以每行拿来计算 （例如：m*n  m是样本个数，n是特征数）
		output = F.softmax(self.output_layer(hidden), dim=1)
		return output


	def predict(self, X):
		return np.argmax(self(X).detach().numpy())


net = Net()

# (8, 3)	8 个样本
data = torch.tensor(np.random.randn(8, 3), dtype=torch.float32)
labels = torch.tensor(np.zeros(8), dtype=torch.int64)		# 要用 (8,) 才行, (1,8)/(8,1) 都不行
# 直接写类别，不能 one-hot
labels[:2] = 0
labels[2:5] = 1
labels[5:] = 2

net.train(data, labels)

data = torch.tensor(np.random.randn(1, 3), dtype=torch.float32)
print('prob:', net(data))
print(net.predict(data))