import sys
sys.path.append("..")
import os
import argparse
import pickle
import logging
import random
import math
import datetime
import time

import torch
import torch.nn as nn
import numpy as np

from utils import Utils
from evaluate import Evaluate


class MF(nn.Module):
	def __init__(self, args, device):
		super(MF, self).__init__()
		self.args = args
		self.device = device

		self.user_params = nn.Parameter(torch.empty(args.max_uid + 1, args.k, dtype=torch.float32, device=device))
		self.item_params = nn.Parameter(torch.empty(args.max_mid + 1, args.k, dtype=torch.float32, device=device))

		nn.init.xavier_normal_(self.user_params)
		nn.init.xavier_normal_(self.item_params)


	def forward(self, x):
		u, i = x[:, -self.args.feature_size].long(), x[:, -(self.args.feature_size - 1)].long()
		return torch.mm(self.user_params[u], self.item_params[i].t())


class Predictor(object):
	def __init__(self, args, predictor, device, mid_map_mfeature, users_has_clicked):
		super(Predictor, self).__init__()
		self.args = args
		self.device = device
		self.predictor = predictor.to(self.device)
		# mid: one-hot feature (21维 -> mid, genre, genre, ...)
		self.mid_map_mfeature = mid_map_mfeature
		self.users_has_clicked = users_has_clicked

		if args.p_optim == 'adam':
			self.optim = torch.optim.Adam(self.predictor.parameters(), lr=args.p_lr, weight_decay=args.weight_decay)
		elif args.p_optim == 'sgd':
			self.optim = torch.optim.SGD(self.predictor.parameters(), lr=args.p_lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif args.p_optim == 'rmsprop':
			self.optim = torch.optim.RMSprop(self.predictor.parameters(), lr=args.p_lr, weight_decay=args.weight_decay)


	def bpr_loss(self, y_ij):
		t = torch.log(torch.sigmoid(y_ij))
		return -torch.sum(t), -t


	def predict(self, data):
		return self.predictor(data)


	def random_negative_sample(self, uid):
		mid = random.randint(0, self.args.max_mid)
		while mid in self.users_has_clicked[uid]:
			mid = random.randint(0, self.args.max_mid)
		return mid


	def train(self, pos_data):
		uids = pos_data[:, -self.args.feature_size].tolist()
		x = pos_data[:, :-self.args.feature_size + 1]	# 除了 mid, genre, genre,..的剩余部分

		neg_mfeature = []
		for uid in uids:
			mid = None
			if self.args.sampler == 'random':
				mid = self.random_negative_sample(uid)
			# TODO 其他采样方式

			mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32, device=self.device)
			neg_mfeature.append(mfeature)

		neg_mfeature = torch.stack(neg_mfeature)
		neg_data = torch.cat([x, neg_mfeature], dim=1)
		y_pos = self.predictor(pos_data)
		y_neg = self.predictor(neg_data)

		y_ij = y_pos - y_neg
		loss, batch_loss = self.bpr_loss(y_ij)
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		return loss, batch_loss


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


class Run(object):
	def __init__(self, args, device, predictor, mid_map_mfeature, users_has_clicked):
		super(Run, self).__init__()
		self.args = args
		self.device = device
		self.predictor = predictor
		self.evaluate = Evaluate(args, device, predictor, users_has_clicked, mid_map_mfeature)

		# mid: one-hot feature (21维 -> mid, genre, genre, ...)
		self.mid_map_mfeature = mid_map_mfeature
		self.users_has_clicked = users_has_clicked
		self.mid_dir = 'without_time_seq/' if args.without_time_seq == 'y' else ''


	def MFtrain(self):
		train_data = np.load(args.base_data_dir + self.mid_dir + 'train_data.npy').tolist()
		loss_list, hr_list, ndcg_list, precs_list = [], [], [], []

		for i_epoch in range(self.args.epoch):
			for i, feature_vector in enumerate(train_data):
				# shape -> (batch, feature size)
				feature_vector = torch.tensor(feature_vector, device=self.device).view(1, -1)
				loss, _ = self.predictor.train(feature_vector.float())

				if (i + 1) % self.args.interval == 0:
					loss_list.append(loss.item())
					info = '{}/{} {}, LOSS:{:.6}'.format(i_epoch + 1, self.args.epoch, i + 1, loss.item())
					print(info)
					logging.info(info)

			hr, ndcg, precs = self.evaluate.evaluate()
			hr_list.append(hr)
			ndcg_list.append(ndcg)
			precs_list.append(precs)
			info = '[Valid]@{} HR{:.6}, NDCG{:.6}, Precs:{:.6}'.format(self.args.topk, hr, ndcg, precs)
			print(info)
			logging.info(info)

		hr, ndcg, precs = self.evaluate.evaluate(title='[TEST]')
		info = '[TEST]@{} HR{:.6}, NDCG{:.6}, Precs:{:.6}'.format(self.args.topk, hr, ndcg, precs)
		print(info)
		logging.info(info)
			


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
	print('device is {}'.format(device))
	logging.info('device is {}'.format(device))

	# mid: one-hot feature (21维 -> mid, genre, genre, ...)
	mid_map_mfeature = Utils.load_obj(args.base_data_dir, 'mid_map_mfeature')
	users_has_clicked = Utils.load_obj(args.base_data_dir, 'users_has_clicked')
	
	model = None
	print('Predictor is {}'.format(args.predictor))
	logging.info('Predictor is {}'.format(args.predictor))
	if args.predictor == 'mf':
		model = MF(args, device)
		args.interval = 1000	# 因为一次只计算一个样本, 输出间隔就放大一点
	elif args.predictor == 'fm':
		pass
	elif args.predictor == 'ncf':
		pass

	predictor = Predictor(args, model, device, mid_map_mfeature, users_has_clicked)
	if args.load_model == 'y':
		pass

	run = Run(args, device, predictor, mid_map_mfeature, users_has_clicked)
	if args.predictor == 'mf':
		run.MFtrain()
	else:
		pass
		# TODO


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--v', default="v")
	parser.add_argument('--num_thread', type=int, default=0)
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_pic_dir', default="pic/")
	parser.add_argument('--base_data_dir', default='../data/ml_1M_row/')
	parser.add_argument('--without_time_seq', default='n')		# 数据集是否按时间排序
	parser.add_argument('--load_model', default='n')			# 是否加载模型
	parser.add_argument('--save_model', default='n')

	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--predictor', default='mf')
	parser.add_argument('--interval', type=int, default=100)
	parser.add_argument('--sampler', default='random')			# 用什么负采样方式
	# embedding
	parser.add_argument('--feature_size', type=int, default=22)	# uid, mid, genres, ...
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=64)
	parser.add_argument('--max_mid', type=int, default=9741)	# 0~9741
	parser.add_argument('--m_emb_dim', type=int, default=64)
	parser.add_argument('--g_emb_dim', type=int, default=32)	# genres emb dim
	# predictor
	parser.add_argument('--p_optim', default='sgd')
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--p_lr', type=float, default=1e-3)
	parser.add_argument('--k', type=int, default=64)	# MF 的隐因子
	parser.add_argument('--epoch', type=int, default=100)
	

	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	main(args, device)