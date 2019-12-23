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
		self.w0 = torch.empty(1, dtype=torch.float32)
		self.w0.requires_grad_(requires_grad=True)
		nn.init.normal_(self.w0)

		self.w1 = torch.empty(feature_size, 1, dtype=torch.float32)
		self.w1.requires_grad_(requires_grad=True)
		nn.init.xavier_normal_(self.w1)

		self.v = torch.empty(feature_size, k, dtype=torch.float32)
		self.v.requires_grad_(requires_grad=True)
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
def evaluate(fm, train_data, target):
	predict = fm(train_data)
	# predict = torch.sigmoid(predict)
	predict_label = torch.zeros(len(target), dtype=torch.int8)
	for i, prob in enumerate(predict):
		if prob > 0.5:
			predict_label[i] = 1
		else:
			predict_label[i] = 0

	target = torch.tensor(target.detach().numpy(), dtype=torch.int8)
	mask = target == predict_label
	precise = torch.sum(mask).float() / len(mask)
	return precise.item()



def trainFM(fm, args, train_data, target):
	# lossFunc = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam([fm.w0, fm.w1, fm.v], lr=args.lr)

	for i_epoch in range(args.epoch):
		predict = fm(train_data)
		# loss = -torch.sum(torch.log(torch.sigmoid(target * predict)))	# 	+ - 1 二分类
		loss = -torch.mean((target*torch.log(predict+1e-8) + (1-target)*torch.log(1 - predict+1e-8)))

		if i_epoch % 500 == 0:
			print('{}/{} LOSS:{:.4f} precise:{:.4f}'.format(args.epoch, i_epoch, 
				loss.item(), evaluate(fm, train_data, target)))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters for FM")
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument('--feature_size', type=int, default=3)
	parser.add_argument('--k', type=int, default=2)
	parser.add_argument('--epoch', type=int, default=10000)
	args = parser.parse_args()

	# movie_embedding_128_mini = load_obj('movie_embedding_128_mini')

	# from loadDataSet import loadData
	# train_data, target = loadData('../../data/testSetRBF2.txt')
	# target = [0 if x == -1 else x for x in target]

	from sklearn.datasets.samples_generator import make_blobs
	train_data, target = make_blobs(n_samples=1000, n_features=3, 
		centers=[[-1,-1,-1], [1,1,1]], cluster_std=[0.4, 0.2], random_state=9)

	train_data = torch.tensor(train_data, dtype=torch.float32)
	target = torch.tensor(target, dtype=torch.float32)

	fm = FM(args.feature_size, args.k)


	trainFM(fm, args, train_data, target)