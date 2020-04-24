import os
import logging
import argparse
import time
import datetime
import random
from collections import deque

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DQN import REM
from Evaluation import Evaluation


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)

class Run(object):
	def __init__(self, args, device):
		super(Run, self).__init__()
		self.args = args
		self.device = device

		self.rem = REM(args, device)
		if args.target == 'y':
			self.target_rem = REM(args, device)
			hard_update(self.target_rem, self.rem)

		self.optim = None
		if args.optim == 'sgd':
			self.optim = torch.optim.SGD(self.rem.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif args.optim == 'rms':
			self.optim = torch.optim.RMSprop(self.rem.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		else:
			self.optim = torch.optim.Adam(self.rem.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, args.lr_decay)
		self.huber_loss = nn.SmoothL1Loss()

		self.evaluate = Evaluation(args, device, self.rem)
		self.build_data_loder()


	def build_data_loder(self):
		state_list = torch.tensor(np.load(self.args.base_data_dir + 'state.npy'), dtype=torch.float32, device=self.device)
		next_state_list = torch.tensor(np.load(self.args.base_data_dir + 'next_state.npy'), dtype=torch.float32, device=self.device)
		action_list = torch.tensor(np.load(self.args.base_data_dir + 'action.npy'), dtype=torch.long, device=self.device)
		dataset = Data.TensorDataset(state_list, next_state_list, action_list)
		self.data_loader = Data.DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=True if self.args.shuffle == 'y' else False)


	def update_parameters(self, data):
		# (batch, 10), (batch, 10), (batch)
		state_batch, next_state_batch, action_batch = data
		q_values = self.rem(state_batch)
		if self.args.target == 'y':
			self.target_rem.eval()
			with torch.no_grad():
				next_q_values = self.target_rem(next_state_batch).detach()
		else:
			next_q_values = self.rem(next_state_batch)

		rewards = torch.tensor([self.args.reward for _ in range(self.args.batch_size)], dtype=torch.float32, device=self.device)
		action_batch = action_batch.view(-1, 1)
		q_values = torch.gather(q_values, 1, action_batch).squeeze()		# (batch)
		next_q_values, _ = next_q_values.max(dim=1)

		target = rewards + self.args.gamma * next_q_values
		loss = self.huber_loss(q_values, target)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		if self.args.target == 'y':
			soft_update(self.target_rem, self.rem, self.args.tau)
		return loss.item()


	def run(self):
		max_ndcg, max_ndcg_epoch = 0, 0
		hr_list, ndcg_list, precs_list = [], [], []
		no_improve_times = 0
		for i_epoch in range(self.args.epoch):
			for i_batch, data in enumerate(self.data_loader):
				self.train()
				loss = self.update_parameters(data)
				if i_batch % 200 == 0:
					loss = round(loss, 5)
					info = f'{i_epoch + 1}/{self.args.epoch}, i_batch:{i_batch}, Loss:{loss}'
					print(info)
					logging.info(info)

			if ((i_epoch + 1) >= self.args.start_eval) and ((i_epoch + 1) % self.args.eval_interval == 0):
				self.eval()
				t1 = time.time()
				with torch.no_grad():
					hr, ndcg, precs = self.evaluate.eval()
				hr, ndcg, precs = round(hr, 5), round(ndcg, 5), round(precs, 5)
				t2 = time.time()
				if ndcg > max_ndcg:
					max_ndcg = ndcg
					max_ndcg_epoch = i_epoch
				else:
					no_improve_times += 1
					if no_improve_times == 2:	# 降低 lr
						self.scheduler.step()
						info = f'LR decay:{self.args.lr_decay}, Optim:{self.optim.lr}'
						print(info)
						logging.info(info)
						no_improve_times = 0

				info = f'[{self.args.mode}]@{self.args.topk} HR:{hr}, NDCG:{ndcg}, Precision:{precs}, Time:{t2 - t1}, Current Max NDCG:{max_ndcg} (epoch:{max_ndcg_epoch})'
				print(info)
				logging.info(info)
				hr_list.append(hr), ndcg_list.append(ndcg), precs_list.append(precs)

			if ((i_epoch + 1) >= self.args.start_save) and ((i_epoch + 1) % self.args.save_interval == 0):
				self.save_model(version=self.args.v, epoch=i_epoch)
				info = f'Saving version:{args.v}_{i_epoch} models'
				print(info)
				logging.info(info)

		self.evaluate.plot_result(precs_list, hr_list, ndcg_list)


	def save_model(self, version, epoch):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		if not os.path.exists('models/' + version + '/'):
			os.makedirs('models/' + version + '/')
		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		torch.save(self.rem.cpu().state_dict(), based_dir + 'REM_' + tail)

	def load_model(self, version, epoch):
		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		self.rem.load_state_dict(torch.load(based_dir + 'REM_' + tail))
		if self.args.target == 'y':
			hard_update(self.target_rem, self.rem)

	def eval(self):
		self.rem.eval()
		if self.args.target == 'y':
			self.target_rem.eval()

	def train(self):
		self.rem.train()


def main(args, device):
	print(f'device:{device}')
	run = Run(args, device)

	# # 加载模型
	if args.load == 'y':
		run.load_model(version=args.load_version, epoch=args.load_epoch)
		info = f'Loading version:{args.load_version}_{args.load_epoch} models'
		print(info)
		logging.info(info)

	run.run()
	if args.save == 'y':
		run.save_model(version=args.v, epoch='final')
		info = f'Saving version:{args.v}_final models'
		print(info)
		logging.info(info)


def init_log(args):
	if not os.path.exists(args.base_log_dir):
		os.makedirs(args.base_log_dir)
	if not os.path.exists(args.base_pic_dir):
		os.makedirs(args.base_pic_dir)
	start = datetime.datetime.now()
	logging.basicConfig(level = logging.INFO,
					filename = args.base_log_dir + args.v + '-' + str(time.time()) + '.log',
					filemode = 'a',
					)
	print('start! '+str(start))
	logging.info('start! '+str(start))
	logging.info('Parameter:')
	logging.info(str(args))
	logging.info('\n-------------------------------------------------------------\n')


if __name__ == '__main__':
	base_data_dir = '../../data/'
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--v', default="v")
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_pic_dir', default="pic/")
	parser.add_argument('--base_data_dir', default=base_data_dir + 'kaggle-RL4REC/')
	parser.add_argument('--shuffle', default='y')
	parser.add_argument('--show', default='n')
	parser.add_argument('--mode', default='valid')		# test/valid
	parser.add_argument('--target', default='y')		# n/y -> target net
	parser.add_argument('--seed', type=int, default=1)

	parser.add_argument('--load', default='n')			# 是否加载模型
	parser.add_argument('--save', default='y')
	parser.add_argument('--load_version', default='v')
	parser.add_argument('--load_epoch', default='final')
	parser.add_argument('--start_save', type=int, default=0)
	parser.add_argument('--save_interval', type=int, default=10)
	parser.add_argument('--start_eval', type=int, default=0)
	parser.add_argument('--eval_interval', type=int, default=10)

	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--reward', type=float, default=1.0)
	parser.add_argument('--topk', type=int, default=10)

	parser.add_argument('--optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--lr_decay', type=float, default=0.5)
	# embedding
	parser.add_argument('--max_iid', type=int, default=70851)	# 0~70851
	parser.add_argument('--i_emb_dim', type=int, default=128)
	# ERM
	parser.add_argument('--K', type=int, default=2)				# 多少个 agent
	parser.add_argument('--seq_hidden_size', type=int, default=128)
	parser.add_argument('--seq_layer_num', type=int, default=1)
	parser.add_argument('--tau', type=float, default=0.9)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--layer_trick', default='none')			# ln/bn/none
	parser.add_argument('--dropout', type=float, default=0.0)

	args = parser.parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# 保持可复现
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)

	init_log(args)	# DEBUG
	main(args, device)