import pickle
import argparse

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# from torchfm.model.fm import FactorizationMachineModel


def save_obj(obj, name):
	with open('../data/ml20/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open('../data/ml20/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


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


# 评估
def evaluate(fm, data, target):
	predict = fm(data)
	rmse = torch.sqrt(torch.sum((target - predict)**2) / predict.shape[0])
	return rmse.item()


def plot_loss_rmse(loss_list, rmse_list, args):
	plt.subplot(1, 3, 1)
	plt.title('FM LOSS')
	plt.xlabel('epoch')
	plt.ylabel('LOSS')
	plt.plot([i for i in range(len(loss_list))], loss_list)

	plt.subplot(1, 3, 3)
	plt.title('FM RMSE')
	plt.xlabel('epoch')
	plt.ylabel('RMSE')
	plt.plot([i for i in range(1, len(loss_list)+1, args.epoch//5)], rmse_list)

	plt.show()


def train(args, model, data, target):
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	loss_list = []
	rmse_list = []

	for i_epoch in range(args.epoch):
		predict = model(data)
		loss = criterion(predict, target)
		loss_list.append(loss.item())

		if i_epoch % (args.epoch//5) == 0:
			rmse = evaluate(model, data, target)
			rmse_list.append(rmse)
			print('{}/{} | LOSS:{:.4f} RMSE:{:.4f}'.format(args.epoch, i_epoch, 
				loss.item(), rmse))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	plot_loss_rmse(loss_list, rmse_list, args)


def normalize(data):
	data[:, 0] = (data[:, 0] - 1) / (610 - 1)
	data[:, 1] = (data[:, 1] - 1) / (117590 - 1)
	return data

def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for FM")
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument('--feature_size', type=int, default=21)
	parser.add_argument('--k', type=int, default=10)
	parser.add_argument('--epoch', type=int, default=1000)
	args = parser.parse_args()
	
	data = np.load('../data/ml20/mini_data.npy').astype(np.float32)
	target = np.load('../data/ml20/mini_target.npy')

	data = normalize(data)
	data = torch.tensor(data, dtype=torch.float32)
	target = torch.tensor(target, dtype=torch.float32).reshape((target.shape[0], 1))

	fm = FM(args.feature_size, args.k)
	train(args, fm, data, target)



if __name__ == '__main__':
	main()