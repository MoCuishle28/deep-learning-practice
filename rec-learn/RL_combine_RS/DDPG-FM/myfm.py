import torch
import torch.nn as nn

# 测试用
import argparse
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt


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


def get_rmse(prediction, target):
	prediction = prediction.squeeze()
	rmse = torch.sqrt(torch.sum((prediction - target)**2) / prediction.shape[0])
	return rmse.item()


def Standardization_uid_mid(data):
	uid_mean = data[:, 0].mean()
	uid_std = data[:, 0].std()
	mid_mean = data[:, 1].mean()
	mid_std = data[:, 1].std()
	data[:, 0] = (data[:, 0] - uid_mean) / uid_std
	data[:, 1] = (data[:, 1] - mid_mean) / mid_std
	return data


def evaluate(predictor, data, target, title='[Valid]'):
	prediction, loss = predictor.train(data, target)
	rmse = get_rmse(prediction, target)
	print(title + ' loss:{:.5}, RMSE:{:.5}'.format(loss.item(), rmse))
	return rmse


def train(args, predictor):
	rmse_list, valid_rmse_list, loss_list = [], [], []

	train_data = torch.tensor(np.load(args.base_data_dir + 'train_data.npy').astype(np.float32), dtype=torch.float32)
	train_target = torch.tensor(np.load(args.base_data_dir + 'train_target.npy').astype(np.float32), dtype=torch.float32)
	valid_data = torch.tensor(np.load(args.base_data_dir + 'valid_data.npy').astype(np.float32), dtype=torch.float32)
	valid_target = torch.tensor(np.load(args.base_data_dir + 'valid_target.npy').astype(np.float32), dtype=torch.float32)
	test_data = torch.tensor(np.load(args.base_data_dir + 'test_data.npy').astype(np.float32), dtype=torch.float32)
	test_target = torch.tensor(np.load(args.base_data_dir + 'test_target.npy').astype(np.float32), dtype=torch.float32)

	train_data = Standardization_uid_mid(train_data)
	valid_data = Standardization_uid_mid(valid_data)
	test_data = Standardization_uid_mid(test_data)

	train_data_set = Data.TensorDataset(train_data, train_target)
	train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=args.batch_size, shuffle=False)

	for epoch in range(args.epoch):
		for i_batch, (data, target) in enumerate(train_data_loader):
			prediction, loss = predictor.train(data, target)
			rmse = get_rmse(prediction, target)
			
			if (i_batch + 1) % 10 == 0:
				print('epoch:{}, i_batch:{}, loss:{:.5}, RMSE:{:.5}'.format(epoch + 1, 
					i_batch+1, loss.item(), rmse))

				rmse_list.append(rmse)
				loss_list.append(loss.item())
		
		valid_rmse = evaluate(predictor, valid_data, valid_target)
		valid_rmse_list.append(valid_rmse)

	evaluate(predictor, test_data, test_target, title='[Test]')
	return rmse_list, valid_rmse_list, loss_list



def plot_result(rmse_list, valid_rmse_list, loss_list):
	plt.figure(figsize=(8, 8))
	plt.subplot(5, 1, 1)
	plt.title('Train RMSE')
	plt.xlabel('Step')
	plt.ylabel('RMSE')
	plt.plot(rmse_list)

	plt.subplot(5, 1, 3)
	plt.title('Valid RMSE')
	plt.xlabel('Step')
	plt.ylabel('RMSE')
	plt.plot(valid_rmse_list)

	plt.subplot(5, 1, 5)
	plt.title('Predictor LOSS')
	plt.xlabel('Step')
	plt.ylabel('LOSS')
	plt.plot(loss_list)

	plt.show()


def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for Predictor")
	parser.add_argument('--v', default="v")
	parser.add_argument('--base_log_dir', default="../data/ddpg-fm/log/")
	parser.add_argument('--base_data_dir', default='../../data/new_ml_1M/')
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--predictor', default='fm')
	# predictor
	parser.add_argument("--predictor_lr", type=float, default=1e-3)
	# FM
	parser.add_argument('--fm_feature_size', type=int, default=22)	# 还要原来基础加上 actor_output
	parser.add_argument('--k', type=int, default=128)
	# network
	parser.add_argument('--hidden_0', type=int, default=128)
	parser.add_argument('--hidden_1', type=int, default=256)
	args = parser.parse_args()

	model = None
	if args.predictor == 'fm':
		print('Predictor is FM.')
		model = FM(args.fm_feature_size, args.k)
	elif args.predictor == 'net':
		print('Predictor is Network.')
		model = Net(args.fm_feature_size, args.hidden_0, args.hidden_1, 1)

	predictor = Predictor(args, model)
	rmse_list, valid_rmse_list, loss_list = train(args, predictor)
	plot_result(rmse_list, valid_rmse_list, loss_list)



if __name__ == '__main__':
	main()