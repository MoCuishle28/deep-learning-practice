import torch
import torch.nn as nn


class FM(nn.Module):
	def __init__(self, feature_size, k):
		super(FM, self).__init__()
		self.w0 = nn.Parameter(torch.empty(1, dtype=torch.float32))
		nn.init.normal_(self.w0)

		# 不加初始化会全 0
		self.w1 = nn.Parameter(torch.empty(feature_size, 1, dtype=torch.float32))
		nn.init.xavier_normal_(self.w1)

		# 不加初始化会全 0
		self.v = nn.Parameter(torch.empty(feature_size, k, dtype=torch.float32))
		nn.init.xavier_normal_(self.v)


	def forward(self, X):
		'''
		X: (batch, feature_size)
		'''
		inter_1 = torch.mm(X, self.v)
		inter_2 = torch.mm((X**2), (self.v**2))
		interaction = (0.5*torch.sum((inter_1**2) - inter_2, dim=1)).reshape(X.shape[0], 1)
		predict = self.w0 + torch.mm(X, self.w1) + interaction
		return predict


class Net(nn.Module):
	def __init__(self, input_num, hidden_num0, hidden_num1, output_num):
		super(Net, self).__init__()
		self.in_layer = nn.Linear(input_num, hidden_num0)
		self.in_norm = nn.LayerNorm(hidden_num0, elementwise_affine=True)

		self.hidden_layer = nn.Linear(hidden_num0, hidden_num1)
		self.hidden_norm = nn.LayerNorm(hidden_num1, elementwise_affine=True)

		self.out_layer = nn.Linear(hidden_num1, output_num)


	def forward(self, x):
		x = self.in_layer(x)
		x = self.in_norm(x)
		x = torch.relu(x)

		x = self.hidden_layer(x)
		x = self.hidden_norm(x)
		x = torch.relu(x)

		x = self.out_layer(x)
		return x


class Predictor(object):
	def __init__(self, args, predictor):
		super(Predictor, self).__init__()
		self.predictor = predictor
		self.optim = torch.optim.Adam(self.predictor.parameters(), lr=args.predictor_lr)
		self.criterion = nn.MSELoss()


	def predict(self, input_data, target):
		target = target.reshape((target.shape[0], 1))
		prediction = self.predictor(input_data)
		loss = self.criterion(prediction, target)
		return prediction, loss


	def train(self, input_data, target):
		prediction = self.predictor(input_data)
		loss = self.criterion(prediction, target.unsqueeze(dim=1))
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		return prediction, loss


	def evaluate(self, data, target):
		pass