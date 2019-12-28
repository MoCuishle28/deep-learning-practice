import pickle
import argparse
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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
		return torch.sigmoid(predict)


# 评估
def evaluate(fm, data, target):
	predict = fm(data)
	# 这样会超内存
	# predict_label = torch.tensor([1 if prob > 0.5 else 0 for prob in predict], dtype=torch.int8)
	predict_label = torch.zeros(len(target), 1, dtype=torch.int8)
	for i, prob in enumerate(predict):
		if prob > 0.5:
			predict_label[i] = 1
		else:
			predict_label[i] = 0

	target = torch.tensor(target.detach().numpy(), dtype=torch.int8)
	precise = torch.sum((target == predict_label)).float() / target.shape[0]
	return precise.item()


def plot_loss_precise(loss_list, valid_precise_list, train_precise_list, args):
	plt.subplot(3, 1, 1)
	plt.title('FM LOSS')
	plt.xlabel('epoch')
	plt.ylabel('LOSS')
	plt.plot([i for i in range(len(loss_list))], loss_list)

	plt.subplot(3, 1, 3)
	plt.title('FM valid Precise')
	plt.xlabel('epoch')
	plt.ylabel('Precise')
	plt.plot([i for i in range(len(valid_precise_list))], valid_precise_list, label='valid precise', color='red')
	plt.plot([i for i in range(len(train_precise_list))], train_precise_list, label='train precise', color='blue')

	plt.show()


def generator(data, target, batch):
	for i in range(0, data.shape[0], batch):
		if i+batch < data.shape[0]:
			yield data[i:i+batch, :], target[i:i+batch]
		else:
			yield data[i:, :], target[i:]


def train_without_batch(args, model, train_data, train_target, valid_data, valid_target):
	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	loss_list = []
	valid_precise_list = []
	train_precise_list = []

	for i_epoch in range(args.epoch):
		predict = model(train_data)
		loss = criterion(predict, train_target)
		loss_list.append(loss.item())

		precise = evaluate(model, valid_data, valid_target)
		train_precise = evaluate(model, train_data, train_target)

		valid_precise_list.append(precise)
		train_precise_list.append(train_precise)

		print('{}/{} | LOSS:{:.4f}  Train Precise:{:.4f}%  Valid Precise:{:.4f}%'.format(args.epoch, i_epoch, 
			loss.item(), train_precise*100, precise*100))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	plot_loss_precise(loss_list, valid_precise_list, train_precise_list, args)


def train_with_batch(args, model, train_data, train_target, valid_data, valid_target):
	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	loss_list = []
	valid_precise_list = []
	train_precise_list = []

	for i_epoch in range(args.epoch):
		i = 0
		for data, target in generator(train_data, train_target, args.batch_size):
			predict = model(data)
			loss = criterion(predict, target)
			loss_list.append(loss.item())

			precise = evaluate(model, valid_data, valid_target)
			train_precise = evaluate(model, data, target)
			i += 1

			valid_precise_list.append(precise)
			train_precise_list.append(train_precise)
			print('{}/{} Step:{} | LOSS:{:.4f}  Train Precise:{:.4f}%  Valid Precise:{:.4f}%'.format(args.epoch, i_epoch, 
				i, loss.item(), train_precise*100, precise*100))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	plot_loss_precise(loss_list, valid_precise_list, train_precise_list, args)


def normalize(data):
	data[:, 0] = (data[:, 0] - 1) / (610 - 1)
	data[:, 1] = (data[:, 1] - 1) / (117590 - 1)
	return data


def divide_data(data, target):
	'''
	8:1:1
	return: train_data, train_target, valid_data, valid_target, test_data, test_target
	'''
	unit = data.shape[0] // 10
	train_size, valid_size, test_size = data.shape[0] - 2*unit, unit, unit
	all_index_list = [i for i in range(data.shape[0])]

	valid_index = random.sample(all_index_list, valid_size)
	sample_set = set(valid_index)
	all_index_list = [i for i in all_index_list if i not in sample_set]		# 删去已经抽出的数据索引
	
	test_index = random.sample(all_index_list, test_size)
	sample_set = set(test_index)
	all_index_list = [i for i in all_index_list if i not in sample_set]

	random.shuffle(all_index_list)
	train_index = all_index_list

	data = data.tolist()
	target = target.tolist()

	train_data, train_target = np.array([data[i] for i in train_index]), np.array([target[i] for i in train_index])
	valid_data, valid_target = np.array([data[i] for i in valid_index]), np.array([target[i] for i in valid_index])
	test_data, test_target = np.array([data[i] for i in test_index]), np.array([target[i] for i in test_index])

	return train_data, train_target, valid_data, valid_target, test_data, test_target


def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for FM")
	parser.add_argument("--lr", type=float, default=1e-2)
	parser.add_argument('--feature_size', type=int, default=21)
	parser.add_argument('--k', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=2048)
	parser.add_argument('--epoch', type=int, default=200)
	args = parser.parse_args()
	
	data = np.load('../data/ml20/mini_data_with_negative.npy').astype(np.float32)
	target = np.load('../data/ml20/mini_target_with_negative.npy')

	data = normalize(data)
	train_data, train_target, valid_data, valid_target, test_data, test_target = divide_data(data, target)
	del data
	del target

	train_data = torch.tensor(train_data, dtype=torch.float32)
	train_target = torch.tensor(train_target, dtype=torch.float32).reshape((train_target.shape[0], 1))
	valid_data = torch.tensor(valid_data, dtype=torch.float32)
	valid_target = torch.tensor(valid_target, dtype=torch.float32).reshape((valid_target.shape[0], 1))
	test_data = torch.tensor(test_data, dtype=torch.float32)
	test_target = torch.tensor(test_target, dtype=torch.float32).reshape((test_target.shape[0], 1))

	fm = FM(args.feature_size, args.k)

	train_with_batch(args, fm, train_data, train_target, valid_data, valid_target)
	# train_without_batch(args, fm, train_data, train_target, valid_data, valid_target)

	test_result = evaluate(fm, test_data, test_target)
	print('Test Precise:{}%'.format(test_result*100))


if __name__ == '__main__':
	main()