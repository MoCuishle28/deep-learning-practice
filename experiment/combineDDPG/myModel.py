import os
import pickle

import torch
import torch.nn as nn

# 测试用
import argparse
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime
import time
import random
import math
import heapq


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
		return predict


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
		return x


def load_obj(name):
	with open('../data/ml_1M_row/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Predictor(object):
	def __init__(self, args, predictor, device, mid_map_mfeature):
		super(Predictor, self).__init__()
		self.args = args
		self.device = device
		# mid: one-hot feature (21维 -> mid, genre, genre, ...)
		self.mid_map_mfeature = mid_map_mfeature
		self.predictor = predictor.to(self.device)

		self.users_has_clicked = load_obj('users_has_clicked')
		if args.predictor_optim == 'adam':
			self.optim = torch.optim.Adam(self.predictor.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)
		elif args.predictor_optim == 'sgd':
			self.optim = torch.optim.SGD(self.predictor.parameters(), lr=args.predictor_lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif args.predictor_optim == 'rmsprop':
			self.optim = torch.optim.RMSprop(self.predictor.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)

		self.criterion = nn.MSELoss()


	def bpr_loss(self, y_ij):
		t = torch.log(torch.sigmoid(y_ij))
		return -torch.mean(t)


	def predict(self, data):
		# TODO 返回什么 reward ?
		return self.predictor(data)


	def negative_sample(self, uid):
		mid = random.randint(0, self.args.max_mid)
		while mid in self.users_has_clicked[uid]:
			mid = random.randint(0, self.args.max_mid)
		return mid


	def train(self, pos_data):
		uids = pos_data[:, -self.args.fm_feature_size].tolist()
		x = pos_data[:, :-self.args.fm_feature_size + 1]	# 除了 mid, genre, genre,..的剩余部分

		neg_mfeature = []
		for uid in uids:
			mid = self.negative_sample(uid)
			mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32).to(self.device)
			neg_mfeature.append(mfeature)

		neg_mfeature = torch.stack(neg_mfeature)
		neg_data = torch.cat([x, neg_mfeature], dim=1)

		y_pos = self.predictor(pos_data)
		y_neg = self.predictor(neg_data)
		y_ij = y_pos - y_neg
		loss = self.bpr_loss(y_ij)
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		return loss


	def on_train(self):
		self.predictor.train()

	def on_eval(self):
		self.predictor.eval()


	def save(self, name):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		torch.save(self.predictor.state_dict(), 'models/p_' + name + '.pkl')

	def load(self, name):
		self.predictor.load_state_dict(torch.load('models/p_' + name + '.pkl'))


class Evaluate(object):
	def __init__(self, args, predictor, train_data, valid_data, test_data, users_has_clicked, mid_map_mfeature, device):
		self.args = args
		self.predictor = predictor
		self.train_data = train_data
		self.valid_data = valid_data
		self.test_data = test_data
		self.users_has_clicked = users_has_clicked
		self.mid_map_mfeature = mid_map_mfeature
		self.device = device

		# 先建立 valid 要忽略的 items set
		self.build_ignore_set(train_data.tolist() + test_data.tolist())


	def get_hr(self, rank_list, gt_item):
		for mid in rank_list:
			if mid == gt_item:
				return 1
		return 0.0


	def get_ndcg(self, rank_list, gt_item):
		for i, mid in enumerate(rank_list):
			if mid == gt_item:
				return math.log(2.0, 2) / math.log(i + 2.0, 2)
		return 0.0


	def get_precs(self, rank_list, gt_item):
		for i, mid in enumerate(rank_list):
			if mid == gt_item:
				return 1.0 / (i + 1.0)
		return 0.0


	def evaluate_for_user(self, uid, mid):
		ret = [0.0, 0.0, 0.0]	# hr, ndcg, precs
		map_items_score = {}	# mid: score

		uid_tensor = torch.tensor([uid], dtype=torch.float32).to(self.device)
		mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32).to(self.device)
		input_vector = torch.cat([uid_tensor, mfeature]).unsqueeze(0).to(self.device)	# 一维
		max_score = self.predictor.predict(input_vector)

		user_ignore_set = self.ignore_set[uid]
		count_larger = 0	# Early stop if there are args.topk items larger than max_score
		early_stop = False
		for i in range(self.args.max_mid + 1):
			if i in user_ignore_set:	# 忽略训练过的 item (以及在验证时忽略测试集,测试时忽略验证集)
				continue
			mfeature = torch.tensor(self.mid_map_mfeature[i].astype(np.float32), dtype=torch.float32).to(self.device)
			input_vector = torch.cat([uid_tensor, mfeature]).unsqueeze(0).to(self.device)
			score = self.predictor.predict(input_vector)
			map_items_score[i] = score
			if score > max_score:
				count_larger += 1
			if count_larger > self.args.topk:
				early_stop = True
				break

		if early_stop == False:
			rank_list = heapq.nlargest(self.args.topk, map_items_score, key=map_items_score.get)
			hr = self.get_hr(rank_list, mid)
			ndcg = self.get_ndcg(rank_list, mid)
			precs = self.get_precs(rank_list, mid)

		return ret


	def evaluate(self, title='[Valid]'):
		dataset = None
		self.hits, self.ndcgs, self.precs = [], [], []

		if title == '[Valid]':
			dataset = self.valid_data.tolist()
		else:	# Testing data set
			dataset = self.test_data.tolist()
			self.build_ignore_set(train_data.tolist() + valid_data.tolist())
			print('Testing...')
		
		for vector in dataset:
			uid = vector[0]
			mid = vector[1]
			ret = self.evaluate_for_user(uid, mid)
			self.hits.append(ret[0])
			self.ndcgs.append(ret[1])
			self.precs.append(ret[2])

		self.hits = torch.tensor(self.hits, dtype=torch.float32).to(self.device)
		self.ndcgs = torch.tensor(self.ndcgs, dtype=torch.float32).to(self.device)
		self.precs = torch.tensor(self.precs, dtype=torch.float32).to(self.device)
		return self.hits.mean().item(), self.ndcgs.mean().item(), self.precs.mean().item()
		

	def build_ignore_set(self, ignore_list):
		# TODO 可是负采样的数据也是经历过训练的啊?
		self.ignore_set = {}	# uid: {mid, mid, ...}	用于 训练/测试 的 items, 在 验证集 评估时忽略 (测试集同理)
		for mfeature in ignore_list:
			uid = mfeature[0]
			mid = mfeature[1]

			if uid in self.ignore_set:
				self.ignore_set[uid].add(mid)
			else:
				self.ignore_set[uid] = set([mid])
		self.ignore_set


def train(args, predictor, mid_map_mfeature, train_data, valid_data, test_data, device, users_has_clicked):
	loss_list = []
	precision_list = []
	hr_list = []
	ndcg_list = []

	train_data_set = Data.TensorDataset(train_data)
	train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=args.batch_size, shuffle=True)
	evaluate = Evaluate(args, predictor, train_data, valid_data, test_data, users_has_clicked, mid_map_mfeature, device)

	for epoch in range(args.epoch):
		predictor.on_train()	# 训练模式
		for i_batch, data in enumerate(train_data_loader):
			data = data[0]
			loss = predictor.train(data)
			
			if (i_batch + 1) % 50 == 0:
				print('epoch:{}/{}, i_batch:{}, BPR LOSS:{:.5}'.format(epoch + 1, args.epoch,
					i_batch+1, loss.item()))
				logging.info('epoch:{}/{}, i_batch:{}, BPR LOSS:{:.5}'.format(epoch + 1, args.epoch,
					i_batch+1, loss.item()))
				loss_list.append(loss.item())

		predictor.on_eval()	# 评估模式
		t1 = time.time()
		hr, ndcg, precs = evaluate.evaluate()
		t2 = time.time()
		print('[Valid]@{} HR:{:.4}, NDCG:{:.4}, Precision:{:.4}, Time:{}'.format(args.topk, hr, ndcg, precs, t2 - t1))
		logging.info('[Valid]@{} HR:{:.4}, NDCG:{:.4}, Precision:{:.4}'.format(args.topk, hr, ndcg, precs))
		hr_list.append(hr)
		ndcg_list.append(ndcg)
		precision_list.append(precs)

	predictor.on_eval()	# 评估模式
	hr, ndcg, precs = evaluate.evaluate(title='[TEST]')

	print('[TEST]@{} HR:{:.4}, NDCG:{:.4}, Precision:{:.4}'.format(args.topk, hr, ndcg, precs))
	logging.info('[TEST]@{} HR:{:.4}, NDCG:{:.4}, Precision:{:.4}'.format(args.topk, hr, ndcg, precs))
	return loss_list, precision_list, hr_list, ndcg_list



def plot_result(args, loss_list, precision_list, hr_list, ndcg_list):
	plt.figure(figsize=(8, 8))
	plt.subplot(1, 7, 1)
	plt.title('Valid Precision')
	plt.xlabel('Step')
	plt.ylabel('Precision')
	plt.plot(precision_list)

	plt.subplot(1, 7, 3)
	plt.title('Valid HR')
	plt.xlabel('Step')
	plt.ylabel('HR')
	plt.plot(hr_list)

	plt.subplot(1, 7, 5)
	plt.title('Valid NDCG')
	plt.xlabel('Step')
	plt.ylabel('LOSS')
	plt.plot(ndcg_list)

	plt.subplot(1, 7, 7)
	plt.title('Training LOSS')
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
	init_log(args)
	print(device)

	model = None
	if args.predictor == 'fm':
		print('Predictor is FM.')
		logging.info('Predictor is FM.')
		model = FM(args.u_emb_dim + args.m_emb_dim + args.g_emb_dim, args.k, args, device, without_rl=True)
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
	mid_dir = 'without_time_seq/' if args.without_time_seq == 'y' else ''
	train_data = torch.tensor(np.load(args.base_data_dir + mid_dir + 'train_data.npy').astype(np.float32), dtype=torch.float32).to(device)
	valid_data = torch.tensor(np.load(args.base_data_dir + mid_dir + 'valid_data.npy').astype(np.float32), dtype=torch.float32).to(device)
	test_data = torch.tensor(np.load(args.base_data_dir + mid_dir + 'test_data.npy').astype(np.float32), dtype=torch.float32).to(device)
	
	mid_map_mfeature = load_obj('mid_map_mfeature')
	predictor = Predictor(args, model, device, mid_map_mfeature)
	loss_list, precision_list, hr_list, ndcg_list = train(args, predictor, mid_map_mfeature, train_data, valid_data, test_data, device, predictor.users_has_clicked)

	plot_result(args, loss_list, precision_list, hr_list, ndcg_list)

	# 保存模型
	if args.save == 'y':
		print('Saving version:{} model'.format(args.v))
		torch.save(model.state_dict(), args.base_log_dir + args.v + '.pkl')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters for Predictor")
	parser.add_argument('--v', default="v")
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--base_log_dir', default="log/myModel/")
	parser.add_argument('--base_pic_dir', default="pic/myModel/")
	parser.add_argument('--base_data_dir', default='../data/ml_1M_row/')
	parser.add_argument('--epoch', type=int, default=20)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--predictor', default='fm')
	parser.add_argument('--predictor_optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)
	parser.add_argument('--init_std', type=float, default=0.01)
	parser.add_argument('--min', type=float, default=0.0)
	parser.add_argument('--max', type=float, default=5.0)
	parser.add_argument('--save', default='n')
	parser.add_argument('--load', default='n')
	parser.add_argument('--show', default='n')	# show pic
	parser.add_argument('--recon', default='n')
	parser.add_argument('--weight_decay', type=float, default=1e-4)		# 正则项
	parser.add_argument('--norm_layer', default='ln')					# bn/ln/none
	parser.add_argument('--early_stop', type=int, default=5)
	parser.add_argument('--without_time_seq', default='n')				# 数据集是否按时间排序
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
	# NCF
	parser.add_argument('--n_act', default='relu')
	parser.add_argument('--layers', default='1024,512')
	parser.add_argument('--actor_output', type=int, default=0)
	parser.add_argument('--dropout', type=float, default=0.0)	# dropout (BN 可以不需要)
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	main(args, device)