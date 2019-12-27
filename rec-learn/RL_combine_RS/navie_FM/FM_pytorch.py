import pickle
import argparse

import torch
import torch.nn as nn
import numpy as np


def save_obj(obj, name):
	with open('../../data/ml20/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open('../../data/ml20/' + name + '.pkl', 'rb') as f:
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
		# return predict
		return torch.sigmoid(predict)


# 测试
def evaluate(fm, data, target):
	predict = fm(data)
	# predict = torch.sigmoid(predict)
	predict_label = torch.zeros(len(target), 1, dtype=torch.int8)
	for i, prob in enumerate(predict):
		if prob > 0.5:
			predict_label[i] = 1
		else:
			predict_label[i] = 0

	target = torch.tensor(target.detach().numpy(), dtype=torch.int8)
	mask = (target == predict_label)
	precise = torch.sum(mask).float() / len(mask)
	return precise.item()



def trainFM(fm, args, train_data, target, valid_data, valid_target):
	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(fm.parameters(), lr=args.lr)

	for i_epoch in range(args.epoch):
		predict = fm(train_data)
		loss = criterion(predict, target)

		if i_epoch % (args.epoch//5) == 0:
			print('{}/{} | LOSS:{:.4f} precise:{:.4f}'.format(args.epoch, i_epoch, 
				loss.item(), evaluate(fm, valid_data, valid_target)))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters for FM")
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument('--feature_size', type=int, default=20)
	parser.add_argument('--k', type=int, default=5)
	parser.add_argument('--epoch', type=int, default=1000)
	parser.add_argument('--n_samples', type=int, default=5000)
	args = parser.parse_args()

	from sklearn.datasets.samples_generator import make_blobs
	train_data, target = make_blobs(n_samples=args.n_samples, n_features=args.feature_size, 
		centers=[[-0.6 for _ in range(args.feature_size)], [0 for _ in range(args.feature_size)]], cluster_std=[0.3, 0.3], random_state=9)

	train_data = torch.tensor(train_data, dtype=torch.float32)
	target = torch.tensor(target, dtype=torch.float32).reshape((args.n_samples, 1))

	valid_data, valid_target = make_blobs(n_samples=args.n_samples//2, n_features=args.feature_size, 
		centers=[[-0.6 for _ in range(args.feature_size)], [0 for _ in range(args.feature_size)]], cluster_std=[0.3, 0.3], random_state=5)

	valid_data = torch.tensor(valid_data, dtype=torch.float32)
	valid_target = torch.tensor(valid_target, dtype=torch.float32).reshape((args.n_samples//2, 1))

	print(train_data.shape, target.shape)

	fm = FM(args.feature_size, args.k)


	trainFM(fm, args, train_data, target, valid_data, valid_target)

	test_data, test_target = make_blobs(n_samples=args.n_samples, n_features=args.feature_size, 
		centers=[[-0.5 for _ in range(args.feature_size)], [0 for _ in range(args.feature_size)]], cluster_std=[0.6, 0.6], random_state=7)

	test_data = torch.tensor(test_data, dtype=torch.float32)
	test_target = torch.tensor(test_target, dtype=torch.float32).reshape((args.n_samples, 1))

	print('test precise:{:.4f}'.format(evaluate(fm, test_data, test_target)))