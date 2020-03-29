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
import torch.utils.data as Data
import numpy as np

from utils import Utils
from evaluate import Evaluate
from sampler import Q_Sampler


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
		u, i = u.view(-1), i.view(-1)
		return torch.sum(self.user_params[u] * self.item_params[i], dim=1)


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
		# sampler
		if args.sampler == 'q':
			self.sampler = Q_Sampler(args, device, users_has_clicked)


	def bpr_loss(self, y_ij):
		t = torch.log(torch.sigmoid(y_ij))
		return -torch.sum(t)


	def predict(self, data):
		return self.predictor(data)


	def random_negative_sample(self, uid):
		mid = random.randint(0, self.args.max_mid)
		while mid in self.users_has_clicked[uid]:
			mid = random.randint(0, self.args.max_mid)
		return mid


	def train(self, pos_data, epsilon):
		x = pos_data[:, :-self.args.feature_size + 1]	# 除了 mid, genre, genre,..的剩余部分
		pos_data_list = pos_data.tolist()

		state_action_pairs = []
		neg_mfeature = []
		for data in pos_data_list:
			uid, pos_mid = int(data[0]), int(data[1])
			mid = None
			if self.args.sampler == 'random':
				mid = self.random_negative_sample(uid)
			elif self.args.sampler == 'q':
				param = torch.cat([self.predictor.user_params[uid], self.predictor.item_params[pos_mid]])
				tenosr_pos_data = torch.tensor(data, dtype=torch.float32, device=self.device).view(-1)
				state = torch.cat([tenosr_pos_data, param]).view(1, -1)	# [uid, mid, genre, ..., params...]

				if random.random() < epsilon:
					mid = self.random_negative_sample(uid)
				else:
					mid = self.sampler.argmax_sample(uid, state) if self.args.sample_method == 'argmax' else self.sampler.softmax_sample(uid, state)
				# shape 都是 (batch, size)
				state_action_pairs.append((state.view(-1), torch.tensor([mid], dtype=torch.float32, device=self.device).view(-1)))

			mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32, device=self.device)
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

		sampler_loss, mean_reward = None, None
		if self.args.sampler == 'q':
			sampler_loss, mean_reward = self.train_sampler(loss, y_ij, state_action_pairs)
		return loss.item(), sampler_loss, mean_reward


	def train_sampler(self, loss, y_ij, state_action_pairs):
		'''
		return: sampler's loss, mean reward
		'''
		margin_list = None
		if self.args.reward == 'loss':
			margin_list = loss.tolist()
		elif self.args.reward == 'dismargin':
			margin_list = y_ij.tolist()

		reward_list = []
		for (state, action), margin in zip(state_action_pairs, margin_list):
			uid, pos_mid = int(state[0].item()), int(state[1].item())
			mfeature = torch.tensor(self.mid_map_mfeature[pos_mid].astype(np.float32), dtype=torch.float32, device=self.device)
			tensor_uid = torch.tensor([uid], dtype=torch.float32, device=self.device)
			param = torch.cat([self.predictor.user_params[uid], self.predictor.item_params[pos_mid]])
			next_state = torch.cat([tensor_uid, mfeature, param]).view(-1)

			if self.args.reward == 'loss':
				reward = torch.tensor([margin], dtype=torch.float32, device=self.device)
			elif self.args.reward == 'dismargin':
				reward = torch.tensor([1 if margin <= 0 else 0], dtype=torch.float32, device=self.device)
			reward_list.append(reward.item())
			# shape 都是 (-1)
			self.sampler.replay_buffer.append((state, action, reward, next_state))
		sampler_loss = self.sampler.train()
		return sampler_loss, sum(reward_list) / len(reward_list)


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
		self.evaluate = Evaluate(args, device, predictor.predictor, users_has_clicked, mid_map_mfeature)

		# mid: one-hot feature (21维 -> mid, genre, genre, ...)
		self.mid_map_mfeature = mid_map_mfeature
		self.users_has_clicked = users_has_clicked
		self.mid_dir = 'without_time_seq/' if args.without_time_seq == 'y' else ''


	def train(self):
		loss_list, hr_list, ndcg_list, precs_list = [], [], [], []
		train_data = torch.tensor(np.load(args.base_data_dir + self.mid_dir + 'train_data.npy').astype(np.float32), dtype=torch.float32, device=self.device)
		train_data_set = Data.TensorDataset(train_data)		
		train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=args.batch_size, shuffle=True)

		for i_epoch in range(self.args.epoch):
			self.predictor.on_train()
			# 计算当前探索率
			epsilon = max(self.args.initial_epsilon * (self.args.num_exploration_episodes - i_epoch) / self.args.num_exploration_episodes, self.args.final_epsilon)
			for i, data in enumerate(train_data_loader):
				data = data[0]
				loss, sampler_loss, mean_reward = self.predictor.train(data, epsilon)

				if (i + 1) % self.args.interval == 0:
					loss_list.append(loss)
					info = f'{i_epoch + 1}/{self.args.epoch} batch:{i + 1}, model Loss:{loss}, sampler Loss:{sampler_loss}, mean reward:{mean_reward}, epsilon:{epsilon}'
					print(info)
					logging.info(info)

			self.predictor.on_eval()
			t1 = time.time()
			hr, ndcg, precs = self.evaluate.evaluate()
			t2 = time.time()
			hr_list.append(hr)
			ndcg_list.append(ndcg)
			precs_list.append(precs)
			info = f'[Valid]@{self.args.topk} HR:{hr}, NDCG:{ndcg}, Precs:{precs}, Time:{t2 - t1}'
			print(info)
			logging.info(info)

		self.predictor.on_eval()
		hr, ndcg, precs = self.evaluate.evaluate(title='[TEST]')
		info = f'[TEST]@{self.args.topk} HR:{hr}, NDCG:{ndcg}, Precs:{precs}, Time:{t2 - t1}'
		print(info)
		logging.info(info)
		self.evaluate.plot_result(self.args, loss_list, precs_list, hr_list, ndcg_list)


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
	info = f'device is {device}'
	print(info)
	logging.info(info)

	# mid: one-hot feature (21维 -> mid, genre, genre, ...)
	mid_map_mfeature = Utils.load_obj(args.base_data_dir, 'mid_map_mfeature')
	users_has_clicked = Utils.load_obj(args.base_data_dir, 'users_has_clicked')
	
	model = None
	info = f'Predictor is {args.predictor}'
	print(info)
	logging.info(info)
	if args.predictor == 'mf':
		model = MF(args, device)
	elif args.predictor == 'fm':
		pass
	elif args.predictor == 'ncf':
		pass

	predictor = Predictor(args, model, device, mid_map_mfeature, users_has_clicked)
	if args.load_model == 'y':
		pass

	run = Run(args, device, predictor, mid_map_mfeature, users_has_clicked)
	run.train()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--v', default="v")
	parser.add_argument('--num_thread', type=int, default=4)
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_pic_dir', default="pic/")
	parser.add_argument('--base_data_dir', default='../data/ml_1M_row/')
	parser.add_argument('--without_time_seq', default='n')		# 数据集是否按时间排序
	parser.add_argument('--load_model', default='n')			# 是否加载模型
	parser.add_argument('--save_model', default='n')
	parser.add_argument('--show', default='n')

	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--predictor', default='mf')
	parser.add_argument('--interval', type=int, default=50)
	parser.add_argument('--sampler', default='q')			# 用什么负采样方式(random/q/...)
	# embedding
	parser.add_argument('--feature_size', type=int, default=22)	# uid, mid, genres, ...
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=64)
	parser.add_argument('--max_mid', type=int, default=9741)	# 0~9741
	parser.add_argument('--m_emb_dim', type=int, default=64)
	parser.add_argument('--g_emb_dim', type=int, default=32)	# genres emb dim
	# predictor
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--p_optim', default='sgd')
	parser.add_argument('--momentum', type=float, default=0.8)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--p_lr', type=float, default=1e-3)
	parser.add_argument('--k', type=int, default=64)			# 隐因子
	# Q sampler
	parser.add_argument('--reward', default='dismargin')		# loss/conmargin/dismargin
	parser.add_argument('--sample_method', default='argmax')	# argmax/softmax

	parser.add_argument('--act', default='elu')
	parser.add_argument('--layers', default='512,256')			# 第一个是输入的维度
	parser.add_argument('--layer_trick', default='ln')			# ln/bn/none
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--q_optim', default='adam')
	parser.add_argument('--q_lr', type=float, default=1e-3)

	parser.add_argument('--update_method', default='hard')		# soft/hard
	parser.add_argument('--ntu', type=float, default=10)		# 10 次更新就替换一次 target
	parser.add_argument('--tau', type=float, default=0.1)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--maxlen', type=int, default=8000)
	# 探索过程所占的episode数量
	parser.add_argument('--num_exploration_episodes', type=int, default=50)
	parser.add_argument('--initial_epsilon', type=float, default=0.8)	# 探索起始时的探索率
	parser.add_argument('--final_epsilon', type=float, default=0.01)	# 探索终止时的探索率
	
	args = parser.parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tmp = "cuda" if torch.cuda.is_available() else "cpu"
	args.num_thread = 0 if tmp == 'cuda' else args.num_thread
	main(args, device)