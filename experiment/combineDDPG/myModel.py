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


def build_ignore_set(ignore_list):
	# TODO 可是负采样的数据也是经历过训练的啊?
	ignore_set = {}	# uid: {mid, mid, ...}	用于 训练/测试 的 items, 在 验证集 评估时忽略 (测试集同理)
	for mfeature in ignore_list:
		uid = mfeature[0]
		mid = mfeature[1]
		if uid in ignore_set:
			ignore_set[uid].add(mid)
		else:
			ignore_set[uid] = set([mid])
	return ignore_set

def count_evaluate_mid(data_list):
	# 获取 验证集/测试集 的 item id
	return list(set([mfeature[1] for mfeature in data_list]))

def get_rec_list(args, ranking_tensor, mids):
	ranking_tensor = ranking_tensor.squeeze()
	score_tensor, idx_tensor = ranking_tensor.topk(args.topk)
	rec_list = [[mids[idx], score] for idx, score in zip(idx_tensor.tolist(), score_tensor.tolist())]
	return rec_list

def get_index(users_has_clicked, uid, rec_list):
	# 计算一些指标: hit, DCG, ...
	has_clicked = users_has_clicked[uid]
	hit = 0
	dcg = 0
	for i, pair in enumerate(rec_list):
		if pair[0] in has_clicked:
			hit += 1
			dcg += (1 / math.log(i + 2, 2))
	return hit, dcg

def get_ndcg(args, dcg_list, like_count_list):
	dcg_tensor = torch.tensor(dcg_list, dtype=torch.float32).to(device)
	idcg_list = []
	for user_like_cnt in like_count_list:
		idcg_list.append(sum([1 / math.log(i + 2, 2) for i in range(min(args.topk, user_like_cnt))]))

	idcg_tensor = torch.tensor(idcg_list, dtype=torch.float32).to(device)
	ndcg = torch.mean(dcg_tensor / idcg_tensor)
	return ndcg


def evaluate(args, mid_map_mfeature, predictor, ignore_set, device, users_has_clicked, evaluate_mids, title='[Valid]'):
	hit_list = []			# 每个 user 的 hit
	like_count_list = []	# 每个用户在 验证集/测试集 中点击的 item 的数量
	dcg_list = []			# 每个用户的 dcg

	for uid in range(1, args.max_uid + 1):
		if uid not in ignore_set:
			continue
		ignore_mids = ignore_set[uid]
		uid_tensor = torch.tensor([uid], dtype=torch.float32).to(device)

		mids = []
		ranking_list = []
		batch_data = []
		like_cnt = 0
		for mid in evaluate_mids:
			if mid in ignore_mids:	# 训练过的数据/测试集的数据 不纳入评价范围
				like_cnt += 1
				continue
			mfeature = torch.tensor(mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32).to(device)
			batch_data.append(torch.cat([uid_tensor, mfeature]).to(device))
			mids.append(mid)
			if len(batch_data) == args.batch_size:	# 每个 batch 计算一次
				batch_data = torch.stack(batch_data).to(device)
				batch_scores = predictor.predict(batch_data)
				ranking_list.extend(batch_scores.tolist())
				batch_data = []

		if batch_data != []:
			batch_data = torch.stack(batch_data).to(device)
			batch_scores = predictor.predict(batch_data)
			ranking_list.extend(batch_scores.tolist())

		rec_list = get_rec_list(args, torch.tensor(ranking_list, dtype=torch.float32).to(device), mids)
		hit, dcg = get_index(users_has_clicked, uid, rec_list)
		hit_list.append(hit)
		dcg_list.append(dcg)
		like_count_list.append(like_cnt)

	hit_tensor = torch.tensor(hit_list, dtype=torch.float32).to(device)
	sum_hit = torch.sum(hit_tensor)
	# 计算 Precision
	precision = sum_hit / torch.tensor([args.topk * len(hit_list)], dtype=torch.float32).to(device)
	# 计算 HR
	like_count_tensor = torch.tensor(like_count_list, dtype=torch.float32).to(device)
	hr = sum_hit / torch.sum(like_count_tensor)
	# 计算 NDCG
	ndcg = get_ndcg(args, dcg_list, like_count_list)

	print('{} @{} Precision:{:.4}, HR:{:.4}, NDCG:{:.4}'.format(title, args.topk, precision.item(), hr.item(), ndcg.item()))
	logging.info('{} @{} Precision:{:.4}, HR:{:.4}, NDCG:{:.4}'.format(title, args.topk, precision.item(), hr.item(), ndcg.item()))
	return precision.item(), hr.item(), ndcg.item()


def train(args, predictor, mid_map_mfeature, train_data, valid_data, test_data, device, users_has_clicked):
	loss_list = []
	precision_list = []
	hr_list = []
	ndcg_list = []

	train_data_set = Data.TensorDataset(train_data)
	train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=args.batch_size, shuffle=True)
	tmp_list = train_data.tolist() + test_data.tolist()
	ignore_set = build_ignore_set(tmp_list)
	evaluate_mids = count_evaluate_mid(valid_data.tolist())

	for epoch in range(args.epoch):
		predictor.on_train()	# 训练模式
		for i_batch, data in enumerate(train_data_loader):
			data = data[0]
			loss = predictor.train(data)
			
			if (i_batch + 1) % 50 == 0:
				print('epoch:{}, i_batch:{}, loss:{:.5}'.format(epoch + 1, 
					i_batch+1, loss.item()))

				loss_list.append(loss.item())
				logging.info('epoch:{}, i_batch:{}, loss:{:.5}'.format(epoch + 1, 
					i_batch+1, loss.item()))

		predictor.on_eval()	# 评估模式
		precision, hr, ndcg = evaluate(args, mid_map_mfeature, predictor, ignore_set, device, users_has_clicked, evaluate_mids)
		precision_list.append(precision)
		hr_list.append(hr)
		ndcg_list.append(ndcg)

	predictor.on_eval()	# 评估模式
	tmp_list = train_data.tolist() + valid_data.tolist()
	ignore_set = build_ignore_set(tmp_list)
	evaluate_mids = count_evaluate_mid(test_data.tolist())
	precision, hr, ndcg = evaluate(args, mid_map_mfeature, predictor, ignore_set, device, users_has_clicked, evaluate_mids, title='[TEST]')
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
	train_data = torch.tensor(np.load(args.base_data_dir + 'without_time_seq/' + 'train_data.npy'), dtype=torch.float32).to(device)
	valid_data = torch.tensor(np.load(args.base_data_dir + 'without_time_seq/' + 'valid_data.npy'), dtype=torch.float32).to(device)
	test_data = torch.tensor(np.load(args.base_data_dir + 'without_time_seq/' + 'test_data.npy'), dtype=torch.float32).to(device)
	
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