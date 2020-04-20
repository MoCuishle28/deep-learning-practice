import torch
import torch.nn as nn
import os
# 测试用
import argparse
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime
import time
import random


class FM(nn.Module):
	def __init__(self, feature_size, k, args, device):
		super(FM, self).__init__()
		self.device = device
		self.args = args
		
		u_matrix, i_matrix = self.one_hot_matrix()
		self.u_embedding = nn.Embedding.from_pretrained(u_matrix)
		self.i_embedding = nn.Embedding.from_pretrained(i_matrix)

		self.w0 = nn.Parameter(torch.empty(1, dtype=torch.float32).to(self.device))

		# 不加初始化会全 0
		self.w1 = nn.Parameter(torch.empty(feature_size, 1, dtype=torch.float32).to(self.device))

		# 不加初始化会全 0
		self.v = nn.Parameter(torch.empty(feature_size, k, dtype=torch.float32).to(self.device))

		nn.init.normal_(self.w0, std=args.init_std)
		nn.init.normal_(self.w1, std=args.init_std)
		nn.init.normal_(self.v, std=args.init_std)


	def one_hot_matrix(self):
		u_matrix = torch.zeros(args.max_uid + 1, args.max_uid + 1).scatter_(1, torch.LongTensor([i for i in range(args.max_uid + 1)]).view(-1, 1), 1)
		i_matrix = torch.zeros(args.max_mid + 1, args.max_mid + 1).scatter_(1, torch.LongTensor([i for i in range(args.max_mid + 1)]).view(-1, 1), 1)
		return u_matrix, i_matrix


	def forward(self, x):
		'''
		x: (batch, feature_size)
		'''
		uids = x[:, -self.args.fm_feature_size]
		iids = x[:, -(self.args.fm_feature_size - 1)]
		genres = x[:, -(self.args.fm_feature_size - 2):]

		with torch.no_grad():
			iemb = self.i_embedding(iids.long().to(self.device))
			uemb = self.u_embedding(uids.long().to(self.device))
		x = torch.cat([uemb, iemb, genres], 1).to(self.device)

		inter_1 = torch.mm(x, self.v)
		inter_2 = torch.mm((x**2), (self.v**2))
		interaction = (0.5*torch.sum((inter_1**2) - inter_2, dim=1)).view(-1, 1)
		predict = self.w0 + torch.mm(x, self.w1) + interaction
		return predict


class MLP(nn.Module):
	def __init__(self, args, input_hidden_size, device):
		super(MLP, self).__init__()
		self.args = args
		self.device = device

		activative_func_dict = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leaky':nn.LeakyReLU(), 
		'selu':nn.SELU(), 'prelu':nn.PReLU(), 'tanh':nn.Tanh()}
		self.activative_func = activative_func_dict.get(args.n_act, nn.ReLU())

		layer_trick = None
		if self.args.norm_layer == 'bn':
			layer_trick = nn.BatchNorm1d
		elif self.args.norm_layer == 'ln':
			layer_trick = nn.LayerNorm

		params = []
		layers = [int(x) for x in args.layers.split(',')]
		self.u_embedding = nn.Embedding(args.max_uid + 1, args.u_emb_dim)
		self.m_embedding = nn.Embedding(args.max_mid + 1, args.m_emb_dim)
		self.g_embedding = nn.Linear(args.fm_feature_size - 2, args.g_emb_dim)

		params.append(nn.Linear(input_hidden_size, layers[0]))
		if layer_trick != None:
			params.append(layer_trick(layers[0]))
		params.append(self.activative_func)
		params.append(nn.Dropout(p=args.dropout))

		for i, num in enumerate(layers[:-1]):
			params.append(nn.Linear(num, layers[i + 1]))
			if layer_trick != None:
				params.append(layer_trick(layers[i + 1]))
			params.append(self.activative_func)
			params.append(nn.Dropout(p=args.dropout))

		params.append(nn.Linear(layers[-1], 1))
		self.mlp = nn.Sequential(*params)


	def forward(self, x):
		uids = x[:, -self.args.fm_feature_size]
		iids = x[:, -(self.args.fm_feature_size - 1)]
		genres = x[:, -(self.args.fm_feature_size - 2):]

		iemb = self.m_embedding(iids.long().to(self.device))
		gemb = self.g_embedding(genres.to(self.device))
		uemb = self.u_embedding(uids.long().to(self.device))

		x = torch.cat([uemb, iemb, gemb], 1).to(self.device)
		x = self.mlp(x)
		return x


class Predictor(object):
	def __init__(self, args, predictor, device):
		super(Predictor, self).__init__()
		self.args = args
		self.device = device
		self.predictor = predictor.to(self.device)
		if args.predictor_optim == 'adam':
			self.optim = torch.optim.Adam(self.predictor.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)
		elif args.predictor_optim == 'sgd':
			self.optim = torch.optim.SGD(self.predictor.parameters(), lr=args.predictor_lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif args.predictor_optim == 'rmsprop':
			self.optim = torch.optim.RMSprop(self.predictor.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)

		self.criterion = nn.MSELoss()


	def predict(self, input_data, target):
		target = target.reshape((target.shape[0], 1)).to(self.device)
		prediction = self.predictor(input_data)
		loss = self.criterion(prediction, target)
		return prediction, loss


	def on_train(self):
		self.predictor.train()


	def on_eval(self):
		self.predictor.eval()


	def train(self, input_data, target):
		prediction = self.predictor(input_data)
		loss = self.criterion(prediction, target.unsqueeze(dim=1))
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		return prediction, loss


	def save(self, version, epoch):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		if not os.path.exists('models/' + version + '/'):
			os.makedirs('models/' + version + '/')

		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		torch.save(self.predictor.cpu().state_dict(), based_dir + 'p_' + tail)

	def load(self, version, epoch):
		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		self.predictor.load_state_dict(torch.load(based_dir + 'p_' + tail, map_location=self.args.device))


def get_rmse(prediction, target):
	if prediction.shape != target.shape:
		prediction = prediction.squeeze()
	rmse = torch.sqrt(((prediction - target)**2).mean())
	return rmse.item()


def get_mae(prediction, target):
	if prediction.shape != target.shape:
		prediction = prediction.squeeze()
	mae = (torch.abs(prediction - target)).mean()
	return mae.item()


def evaluate(predictor, data, target, title='[Valid]'):
	prediction, loss = predictor.predict(data, target)
	rmse = get_rmse(prediction, target)
	mae = get_mae(prediction, target)
	return rmse, mae, loss.item()


def train(args, predictor, train_data, train_target, valid_data, valid_target, device):
	rmse_list, valid_rmse_list, loss_list = [], [], []
	min_rmse, min_rmse_epoch = 999, 0

	if args.mode == 'test':
		test_data = torch.tensor(np.load(args.base_log_dir + 'data/' + 'test_data.npy'), dtype=torch.float32).to(device)
		test_target = torch.tensor(np.load(args.base_log_dir + 'data/' + 'test_target.npy'), dtype=torch.float32).to(device)
		train_data = torch.cat([train_data, valid_data], dim=0)
		train_target = torch.cat([train_target, valid_target], dim=0)
	else:	# valid
		test_data = valid_data
		test_target = valid_target

	train_data_set = Data.TensorDataset(train_data, train_target)
	train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=args.batch_size)

	for epoch in range(args.epoch):
		predictor.on_train()	# 训练模式
		for i_batch, (data, target) in enumerate(train_data_loader):
			prediction, loss = predictor.train(data, target)

			with torch.no_grad():
				rmse = get_rmse(prediction, target)
				mae = get_mae(prediction, target)
			
			if (i_batch + 1) % 50 == 0:
				info = 'epoch:{}/{}, i_batch:{}, loss:{:.5}, RMSE:{:.5}, MAE:{:.5}'.format(epoch + 1, args.epoch, i_batch+1, loss.item(), rmse, mae)
				print(info)
				logging.info(info)
				rmse_list.append(rmse)
				loss_list.append(loss.item())

		predictor.on_eval()	# 评估模式
		with torch.no_grad():
			valid_rmse, mae, loss = evaluate(predictor, test_data, test_target)
		valid_rmse, mae, loss = round(valid_rmse, 5), round(mae,5), round(loss, 5)
		valid_rmse_list.append(valid_rmse)
		if valid_rmse < min_rmse:
			min_rmse = valid_rmse
			min_rmse_epoch = epoch
		info = f'mode:{args.mode}, LOSS:{loss}, RMSE:{valid_rmse}, MAE:{mae}, Current Min RMSE:{min_rmse} (in epoch:{min_rmse_epoch})'
		print(info)
		logging.info(info)

		if valid_rmse >= min_rmse and (epoch - min_rmse_epoch + 1) >= args.early_stop:	# early stop
			break

	test_data = torch.tensor(np.load(args.base_log_dir + 'data/' + 'test_data.npy'), dtype=torch.float32).to(device)
	test_target = torch.tensor(np.load(args.base_log_dir + 'data/' + 'test_target.npy'), dtype=torch.float32).to(device)
	with torch.no_grad():
		valid_rmse, mae, loss = evaluate(predictor, test_data, test_target)
	valid_rmse, mae, loss = round(valid_rmse, 5), round(mae,5), round(loss, 5)
	info = f'[TEST], LOSS:{loss}, RMSE:{valid_rmse}, MAE:{mae}'
	print(info)
	logging.info(info)
	return rmse_list, valid_rmse_list, loss_list


def plot_result(args, rmse_list, valid_rmse_list, loss_list):
	plt.figure(figsize=(8, 8))
	plt.subplot(1, 5, 1)
	plt.title('Training RMSE')
	plt.xlabel('Step')
	plt.ylabel('RMSE')
	plt.plot(rmse_list)

	plt.subplot(1, 5, 3)
	plt.title('Testing RMSE')
	plt.xlabel('Step')
	plt.ylabel('RMSE')
	plt.plot(valid_rmse_list)

	plt.subplot(1, 5, 5)
	plt.title('LOSS')
	plt.xlabel('Step')
	plt.ylabel('LOSS')
	plt.plot(loss_list)

	plt.savefig(args.base_pic_dir + args.v + '.png')
	if args.show == 'y':
		plt.show()


def init_log(args):
	start = datetime.datetime.now()
	logging.basicConfig(level = logging.INFO,
					filename = args.base_log_dir + args.v + '-' + str(time.time()) + '.log',
					filemode = 'a',		# 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
					# a是追加模式，默认如果不写的话，就是追加模式
					)
	logging.info('start! '+str(start))
	logging.info('Parameter:')
	logging.info(str(args))
	logging.info('\n-------------------------------------------------------------\n')


def main(args, device):
	model = None
	if args.predictor == 'fm':
		print('Predictor is FM.')
		logging.info('Predictor is FM.')
		model = FM(args.max_uid + 1 + args.max_mid + 1 + 20, args.k, args, device)
	elif args.predictor == 'mlp':
		print('Predictor is MLP.')
		logging.info('Predictor is MLP.')
		model = MLP(args, args.u_emb_dim + args.m_emb_dim + args.g_emb_dim, device)

	# 加载模型
	if args.load == 'y':
		print('Loading version:{} model'.format(args.v))
		model.load_state_dict(torch.load(args.base_log_dir + args.v + '.pkl'))

	# [uid, mid, genres]
	train_data = torch.tensor(np.load(args.base_log_dir + 'data/' + 'train_data.npy'), dtype=torch.float32).to(device)
	train_target = torch.tensor(np.load(args.base_log_dir + 'data/' + 'train_target.npy'), dtype=torch.float32).to(device)
	valid_data = torch.tensor(np.load(args.base_log_dir + 'data/' + 'valid_data.npy'), dtype=torch.float32).to(device)
	valid_target = torch.tensor(np.load(args.base_log_dir + 'data/' + 'valid_target.npy'), dtype=torch.float32).to(device)

	predictor = Predictor(args, model, device)
	rmse_list, valid_rmse_list, loss_list = train(args, predictor, train_data, train_target, valid_data, valid_target, device)

	plot_result(args, rmse_list, valid_rmse_list, loss_list)

	# 保存模型
	if args.save == 'y':
		print('Saving version:{} model'.format(args.v))
		torch.save(model.cpu().state_dict(), args.base_log_dir + args.v + '.pkl')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters for Predictor")
	parser.add_argument('--v', default="v")
	parser.add_argument('--mode', default="valid")
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--base_log_dir', default="../data/ddpg-fm/traditional-model/")
	parser.add_argument('--base_pic_dir', default="../data/ddpg-fm/traditional-model/")
	parser.add_argument('--base_data_dir', default='../../data/new_ml_1M/')
	parser.add_argument('--epoch', type=int, default=20)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--predictor', default='fm')
	parser.add_argument('--predictor_optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.0)
	parser.add_argument('--weight_decay', type=float, default=1e-2)
	parser.add_argument('--init_std', type=float, default=0.01)
	parser.add_argument('--save', default='n')
	parser.add_argument('--load', default='n')
	parser.add_argument('--show', default='n')	# show pic
	parser.add_argument('--norm_layer', default='none')	# bn/ln/none
	parser.add_argument('--early_stop', type=int, default=3)
	# predictor
	parser.add_argument("--predictor_lr", type=float, default=1e-2)
	# FM
	parser.add_argument('--fm_feature_size', type=int, default=22)
	parser.add_argument('--k', type=int, default=8)
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--max_mid', type=int, default=9741)	# 0~9741
	# MLP
	parser.add_argument('--layers', default='1024,512,256')
	parser.add_argument('--n_act', default='relu')
	parser.add_argument('--u_emb_dim', type=int, default=128)
	parser.add_argument('--m_emb_dim', type=int, default=128)
	parser.add_argument('--g_emb_dim', type=int, default=16)	# genres emb dim
	parser.add_argument('--dropout', type=float, default=0.0)	# dropout (BN 可以不需要)
	args = parser.parse_args()

	init_log(args)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)

	main(args, device)