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


class FM(nn.Module):
	def __init__(self, feature_size, k, args, device, without_rl=False):
		super(FM, self).__init__()
		self.device = device
		self.args = args
		self.without_rl = without_rl

		
		self.u_embedding = nn.Embedding(args.max_uid + 1, args.u_emb_dim)
		self.m_embedding = nn.Embedding(args.max_mid + 1, args.m_emb_dim)
		self.g_embedding = nn.Linear(args.fm_feature_size - 2, args.g_emb_dim)

		self.w0 = nn.Parameter(torch.empty(1, dtype=torch.float32).to(self.device))

		# 不加初始化会全 0
		self.w1 = nn.Parameter(torch.empty(feature_size, 1, dtype=torch.float32).to(self.device))

		# 不加初始化会全 0
		self.v = nn.Parameter(torch.empty(feature_size, k, dtype=torch.float32).to(self.device))

		nn.init.normal_(self.w0, std=args.init_std)
		nn.init.xavier_normal_(self.w1)
		nn.init.xavier_normal_(self.v)


	def forward(self, x):
		'''
		x: (batch, feature_size)
		'''
		mids = x[:, -(self.args.fm_feature_size - 1)]
		genres = x[:, -(self.args.fm_feature_size - 2):]

		memb = self.m_embedding(mids.long().to(self.device))
		gemb = self.g_embedding(genres.to(self.device))

		uids = x[:, -self.args.fm_feature_size]
		uemb = self.u_embedding(uids.long().to(self.device))

		if self.without_rl:
			x = torch.cat([uemb, memb, gemb], 1).to(self.device)
		else:
			x = x[:, :-self.args.fm_feature_size]	# 有 RL 部分的输出作为 user embedding
			x = torch.cat([x, uemb, memb, gemb], 1).to(self.device)

		inter_1 = torch.mm(x, self.v)
		inter_2 = torch.mm((x**2), (self.v**2))
		interaction = (0.5*torch.sum((inter_1**2) - inter_2, dim=1)).reshape(x.shape[0], 1)
		predict = self.w0 + torch.mm(x, self.w1) + interaction
		return predict.clamp(min=self.args.min, max=self.args.max)


class Net(nn.Module):
	def __init__(self, input_num, hidden_num0, hidden_num1, output_num, args, device, without_rl=False):
		super(Net, self).__init__()
		self.args = args
		self.device = device
		self.without_rl = without_rl

		activative_func_dict = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leaky':nn.LeakyReLU(), 
		'selu':nn.SELU(), 'prelu':nn.PReLU(), 'tanh':nn.Tanh()}
		self.activative_func = activative_func_dict.get(args.n_act, nn.ReLU())
		# embedding
		self.u_embedding = nn.Embedding(args.max_uid + 1, args.u_emb_dim)
		self.m_embedding = nn.Embedding(args.max_mid + 1, args.m_emb_dim)
		self.g_embedding = nn.Linear(args.fm_feature_size - 2, args.g_emb_dim)

		self.in_layer = nn.Linear(input_num, hidden_num0)
		self.hidden_layer = nn.Linear(hidden_num0, hidden_num1)
		self.out_layer = nn.Linear(hidden_num1, output_num)

		if self.args.norm_layer == 'bn':
			self.in_norm = nn.BatchNorm1d(hidden_num0, affine=True)
			self.hidden_norm = nn.BatchNorm1d(hidden_num1, affine=True)
		elif self.args.norm_layer == 'ln':
			self.in_norm = nn.LayerNorm(hidden_num0, elementwise_affine=True)
			self.hidden_norm = nn.LayerNorm(hidden_num1, elementwise_affine=True)


	def forward(self, x):
		mids = x[:, -(self.args.fm_feature_size - 1)]
		genres = x[:, -(self.args.fm_feature_size - 2):]

		memb = self.m_embedding(mids.long().to(self.device))
		gemb = self.g_embedding(genres.to(self.device))

		uids = x[:, -self.args.fm_feature_size]
		uemb = self.u_embedding(uids.long().to(self.device))

		if self.without_rl:
			x = torch.cat([uemb, memb, gemb], 1).to(self.device)
		else:
			x = x[:, :-self.args.fm_feature_size]
			x = torch.cat([x, uemb, memb, gemb], 1).to(self.device)

		x = self.in_layer(x)
		x = self.in_norm(x) if self.args.norm_layer != 'none' else x
		x = self.activative_func(x)

		x = self.hidden_layer(x)
		x = self.hidden_norm(x) if self.args.norm_layer != 'none' else x
		x = self.activative_func(x)

		x = self.out_layer(x)
		return x.clamp(min=self.args.min, max=self.args.max)


class NCF(nn.Module):
	def __init__(self, args, input_hidden_size, device, without_rl=False):
		super(NCF, self).__init__()
		self.args = args
		self.device = device
		self.without_rl = without_rl

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
		self.ncf = nn.Sequential(*params)


	def forward(self, x):
		mids = x[:, -(self.args.fm_feature_size - 1)]
		genres = x[:, -(self.args.fm_feature_size - 2):]

		memb = self.m_embedding(mids.long().to(self.device))
		gemb = self.g_embedding(genres.to(self.device))

		uids = x[:, -self.args.fm_feature_size]
		uemb = self.u_embedding(uids.long().to(self.device))

		if self.without_rl:
			x = torch.cat([uemb, memb, gemb], 1).to(self.device)
		else:
			x = x[:, :-self.args.fm_feature_size]
			x = torch.cat([x, uemb, memb, gemb], 1)
		x = self.ncf(x)
		return x.clamp(min=self.args.min, max=self.args.max)


class Predictor(object):
	def __init__(self, args, predictor, device):
		super(Predictor, self).__init__()
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
		torch.save(self.predictor.state_dict(), based_dir + 'p_' + tail)

	def load(self, version, epoch):
		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		self.predictor.load_state_dict(torch.load(based_dir + 'p_' + tail))


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
	prediction, loss = predictor.predict(data, target)
	rmse = get_rmse(prediction, target)
	print(title + ' loss:{:.5}, RMSE:{:.5}'.format(loss.item(), rmse))
	logging.info(title + ' loss:{:.5}, RMSE:{:.5}'.format(loss.item(), rmse))
	return rmse


def train(args, predictor, train_data, train_target, valid_data, valid_target, device):
	rmse_list, valid_rmse_list, loss_list = [], [], []
	pre_rmse = 99999
	increase_time = 0	# 连续 x 次不下降

	train_data_set = Data.TensorDataset(train_data, train_target)
	train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=args.batch_size, shuffle=True)

	for epoch in range(args.epoch):
		predictor.on_train()	# 训练模式
		for i_batch, (data, target) in enumerate(train_data_loader):
			prediction, loss = predictor.train(data, target)
			rmse = get_rmse(prediction, target)
			
			if (i_batch + 1) % 50 == 0:
				print('epoch:{}, i_batch:{}, loss:{:.5}, RMSE:{:.5}'.format(epoch + 1, 
					i_batch+1, loss.item(), rmse))

				rmse_list.append(rmse)
				loss_list.append(loss.item())
				logging.info('epoch:{}, i_batch:{}, loss:{:.5}, RMSE:{:.5}'.format(epoch + 1, 
					i_batch+1, loss.item(), rmse))

		predictor.on_eval()	# 评估模式
		valid_rmse = evaluate(predictor, valid_data, valid_target)
		if valid_rmse >= pre_rmse:	# early stop
			increase_time += 1
		else:
			increase_time = 0
		pre_rmse = valid_rmse
		valid_rmse_list.append(valid_rmse)
		if increase_time >= args.early_stop:
			break

	predictor.on_eval()	# 评估模式
	test_data = torch.tensor(np.load(args.base_log_dir + 'data/' + 'test_data.npy'), dtype=torch.float32).to(device)
	test_target = torch.tensor(np.load(args.base_log_dir + 'data/' + 'test_target.npy'), dtype=torch.float32).to(device)
	evaluate(predictor, test_data, test_target, title='[Test]')
	return rmse_list, valid_rmse_list, loss_list



def plot_result(args, rmse_list, valid_rmse_list, loss_list):
	plt.figure(figsize=(8, 8))
	plt.subplot(1, 5, 1)
	plt.title('Train RMSE')
	plt.xlabel('Step')
	plt.ylabel('RMSE')
	plt.plot(rmse_list)

	plt.subplot(1, 5, 3)
	plt.title('Valid RMSE')
	plt.xlabel('Step')
	plt.ylabel('RMSE')
	plt.plot(valid_rmse_list)

	plt.subplot(1, 5, 5)
	plt.title('Predictor LOSS')
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


def data_reconstruct(args, device):
	train_data = torch.tensor(np.load(args.base_data_dir + 'train_data.npy').astype(np.float32), dtype=torch.float32).to(device)
	train_target = torch.tensor(np.load(args.base_data_dir + 'train_target.npy').astype(np.float32), dtype=torch.float32).to(device)
	valid_data = torch.tensor(np.load(args.base_data_dir + 'valid_data.npy').astype(np.float32), dtype=torch.float32).to(device)
	valid_target = torch.tensor(np.load(args.base_data_dir + 'valid_target.npy').astype(np.float32), dtype=torch.float32).to(device)
	test_data = torch.tensor(np.load(args.base_data_dir + 'test_data.npy').astype(np.float32), dtype=torch.float32).to(device)
	test_target = torch.tensor(np.load(args.base_data_dir + 'test_target.npy').astype(np.float32), dtype=torch.float32).to(device)

	data = torch.cat([train_data, valid_data, test_data]).to(device)
	target = torch.cat([train_target, valid_target, test_target]).to(device)
	all_data = Data.TensorDataset(data, target)

	train_size, valid_size = train_data.shape[0], valid_data.shape[0]
	train_data, valid_data, test_data = torch.utils.data.random_split(all_data, [train_size, valid_size, valid_size])

	train_data = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
	valid_data = Data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=True)
	test_data = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)

	d = {'train':train_data, 'valid':valid_data, 'test':test_data}
	for k, v in d.items():
		data_list = []
		target_list = []
		for data, target in v:
			data_list.append(data.numpy())
			target_list.append(target.numpy())
		data = np.concatenate(data_list)
		target = np.concatenate(target_list)
		np.save(args.base_log_dir + 'data/' + k + '_data.npy', data)
		np.save(args.base_log_dir + 'data/' + k + '_target.npy', target)
		print(data.shape, target.shape)


def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for Predictor")
	parser.add_argument('--v', default="v")
	parser.add_argument('--base_log_dir', default="../data/ddpg-fm/traditional-model/")
	parser.add_argument('--base_pic_dir', default="../data/ddpg-fm/traditional-model/")
	parser.add_argument('--base_data_dir', default='../../data/new_ml_1M/')
	parser.add_argument('--epoch', type=int, default=20)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--predictor', default='ncf')
	parser.add_argument('--predictor_optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--init', default='normal')
	parser.add_argument('--kaiming_mode', default='fan_in')
	parser.add_argument('--kaiming_func', default='relu')
	parser.add_argument('--init_std', type=float, default=0.01)
	parser.add_argument('--min', type=float, default=0.0)
	parser.add_argument('--max', type=float, default=5.0)
	parser.add_argument('--save', default='n')
	parser.add_argument('--load', default='n')
	parser.add_argument('--show', default='n')	# show pic
	parser.add_argument('--recon', default='n')
	parser.add_argument('--norm_layer', default='bn')	# bn/ln/none
	parser.add_argument('--early_stop', type=int, default=5)
	# predictor
	parser.add_argument("--predictor_lr", type=float, default=1e-3)
	# FM
	parser.add_argument('--fm_feature_size', type=int, default=22)	# 还要原来基础加上 actor_output
	parser.add_argument('--k', type=int, default=8)
	# network
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=128)
	parser.add_argument('--max_mid', type=int, default=9741)	# 0~9741
	parser.add_argument('--m_emb_dim', type=int, default=128)
	parser.add_argument('--g_emb_dim', type=int, default=16)	# genres emb dim

	parser.add_argument('--n_act', default='relu')
	parser.add_argument('--hidden_0', type=int, default=1024)
	parser.add_argument('--hidden_1', type=int, default=512)
	# NCF
	parser.add_argument('--layers', default='1024,512,256')
	parser.add_argument('--actor_output', type=int, default=0)
	parser.add_argument('--dropout', type=float, default=0.0)	# dropout (BN 可以不需要)
	args = parser.parse_args()

	init_log(args)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	model = None
	if args.predictor == 'fm':
		print('Predictor is FM.')
		logging.info('Predictor is FM.')
		model = FM(args.u_emb_dim + args.m_emb_dim + args.g_emb_dim, args.k, args, device, without_rl=True)
	elif args.predictor == 'net':
		print('Predictor is Network.')
		logging.info('Predictor is Network.')
		model = Net(args.u_emb_dim + args.m_emb_dim + args.g_emb_dim, args.hidden_0, args.hidden_1, 1, args, device, without_rl=True)
	elif args.predictor == 'ncf':
		print('Predictor is NCF.')
		logging.info('Predictor is NCF.')
		model = NCF(args, args.u_emb_dim + args.m_emb_dim + args.g_emb_dim, device, without_rl=True)

	# 加载模型
	if args.load == 'y':
		print('Loading version:{} model'.format(args.v))
		model.load_state_dict(torch.load(args.base_log_dir + args.v + '.pkl'))

	if args.recon == 'y':	# 打乱重构数据, 防止出现先验概率误差太大问题
		data_reconstruct(args)
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
		torch.save(model.state_dict(), args.base_log_dir + args.v + '.pkl')


if __name__ == '__main__':
	main()