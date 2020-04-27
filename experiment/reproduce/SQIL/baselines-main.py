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

from baselines import GRU


class Evaluation(object):
	def __init__(self, args, device, agent):
		super(Evaluation, self).__init__()
		self.args = args
		self.device = device
		self.agent = agent
		
		if args.mode == 'valid':
			self.eval_sessions = pd.read_pickle(args.base_data_dir + 'sampled_val.df')
		elif args.mode == 'test':
			self.eval_sessions = pd.read_pickle(args.base_data_dir + 'sampled_test.df')


	def compute_index(self, state_batch, action_batch):
		state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
		prediction_batch = self.agent(state_batch)
		for prediction, action in zip(prediction_batch, action_batch):
			_, rec_iids = prediction.topk(self.args.topk)
			rec_iids = rec_iids.view(-1).tolist()
			hr, ndcg, precs = self.get_hr(rec_iids, action), self.get_ndcg(rec_iids, action), self.get_precs(rec_iids, action)
			self.hr_list.append(hr), self.ndcg_list.append(ndcg), self.precs_list.append(precs)


	def eval(self):
		eval_ids = self.eval_sessions.session_id.unique()
		groups = self.eval_sessions.groupby('session_id')
		self.hr_list, self.ndcg_list, self.precs_list = [], [], []
		state_batch, action_batch = [], []
		for sid in eval_ids:
			group = groups.get_group(sid)
			history = []
			for index, row in group.iterrows():
				if history == []:
					history.append(row['item_id'])
					continue
				state = list(history)
				state = self.pad_history(state, 10, self.args.max_iid)
				state_batch.append(state)
				action_batch.append(row['item_id'])
				history.append(row['item_id'])

			if len(state_batch) >= self.args.batch_size:
				self.compute_index(state_batch, action_batch)
				state_batch, action_batch = [], []
		
		if len(state_batch)	!= 0:
			self.compute_index(state_batch, action_batch)
		hr = torch.tensor(self.hr_list, dtype=torch.float32, device=self.device)
		ndcg = torch.tensor(self.ndcg_list, dtype=torch.float32, device=self.device)
		precs = torch.tensor(self.precs_list, dtype=torch.float32, device=self.device)
		return hr.mean().item(), ndcg.mean().item(), precs.mean().item()


	def pad_history(self, itemlist, length, pad_item):
		if len(itemlist) >= length:
			return itemlist[-length:]
		if len(itemlist) < length:
			temp = [pad_item] * (length - len(itemlist))
			itemlist.extend(temp)
			return itemlist


	def get_hr(self, rank_list, gt_item):
		for iid in rank_list:
			if iid == gt_item:
				return 1
		return 0.0

	def get_ndcg(self, rank_list, gt_item):
		for i, iid in enumerate(rank_list):
			if iid == gt_item:
				return (np.log(2.0) / np.log(i + 2.0)).item()
		return 0.0

	def get_precs(self, rank_list, gt_item):
		for i, iid in enumerate(rank_list):
			if iid == gt_item:
				return 1.0 / (i + 1.0)
		return 0.0

	def plot_result(self, precision_list, hr_list, ndcg_list):
		plt.figure(figsize=(8, 8))
		plt.subplot(1, 5, 1)
		plt.title('Valid Precision')
		plt.xlabel('Step')
		plt.ylabel('Precision')
		plt.plot(precision_list)

		plt.subplot(1, 5, 3)
		plt.title('Valid HR')
		plt.xlabel('Step')
		plt.ylabel('HR')
		plt.plot(hr_list)

		plt.subplot(1, 5, 5)
		plt.title('Valid NDCG')
		plt.xlabel('Step')
		plt.ylabel('LOSS')
		plt.plot(ndcg_list)

		plt.savefig(self.args.base_log_dir + self.args.v + '.png')
		if self.args.show == 'y':
			plt.show()


def get_optim(key, agent, lr, args):
	optim = None
	if key == 'adam':
		optim = torch.optim.Adam(agent.parameters(), lr=lr, weight_decay=args.weight_decay)
	elif key == 'sgd':
		optim = torch.optim.SGD(agent.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
	elif key == 'rms':
		optim = torch.optim.RMSprop(agent.parameters(), lr=lr, weight_decay=args.weight_decay)
	return optim


class Run(object):
	def __init__(self, args, device):
		super(Run, self).__init__()
		self.args = args
		self.device = device

		if args.model == 'gru':
			self.model = GRU(args, device).to(device)
		else:
			pass

		self.optim = get_optim(args.optim, self.model, args.lr, args)
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, args.lr_decay)
		self.loss_func = nn.CrossEntropyLoss()
		self.evaluate = Evaluation(args, device, self.model)
		self.build_data_loder()

	def build_data_loder(self):
		state_list = torch.tensor(np.load(self.args.base_data_dir + 'state.npy'), dtype=torch.float32, device=self.device)
		action_list = torch.tensor(np.load(self.args.base_data_dir + 'action.npy'), dtype=torch.long, device=self.device)
		dataset = Data.TensorDataset(state_list, action_list)
		self.data_loader = Data.DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=True if self.args.shuffle == 'y' else False)


	def run(self):
		max_ndcg, max_ndcg_epoch = 0, 0
		hr_list, ndcg_list, precs_list = [], [], []
		no_improve_times = 0
		for i_epoch in range(self.args.epoch):
			for i_batch, (state, target) in enumerate(self.data_loader):
				self.model.train()
				prediction = self.model(state)
				loss = self.loss_func(prediction, target)

				self.optim.zero_grad()
				loss.backward()
				self.optim.step()

				if i_batch % 200 == 0:
					loss = round(loss.item(), 5)
					info = f'{i_epoch + 1}/{self.args.epoch}, batch:{i_batch}, LOSS:{loss}'
					print(info)
					logging.info(info)

			if ((i_epoch + 1) >= self.args.start_eval) and ((i_epoch + 1) % self.args.eval_interval == 0):
				self.model.eval()
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
						info = f'LR decay:{self.args.lr_decay}, Optim:{self.optim}'
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
		if not os.path.exists('models/baselines_' + version + '/'):
			os.makedirs('models/baselines_' + version + '/')

		based_dir = 'models/baselines_' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		torch.save(self.model.state_dict(), f'{based_dir}{self.args.model}_{tail}')

	def load_model(self, version, epoch):
		based_dir = 'models/baselines_' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		self.model.load_state_dict(torch.load(f'{based_dir}{self.args.model}_{tail}'))


def main(args, device):
	print(f'device:{device}')
	run = Run(args, device)

	# 加载模型
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
	parser.add_argument('--model', default="gru")
	parser.add_argument('--base_log_dir', default="log-baseline/")
	parser.add_argument('--base_data_dir', default=base_data_dir + 'kaggle-RL4REC/')
	parser.add_argument('--shuffle', default='y')
	parser.add_argument('--show', default='n')
	parser.add_argument('--mode', default='valid')		# test/valid
	parser.add_argument('--target', default='n')		# n/y -> target net
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
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=512)

	parser.add_argument('--optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--lr_decay', type=float, default=0.5)
	# embedding
	parser.add_argument('--max_iid', type=int, default=70851)	# 0~70851
	parser.add_argument('--m_emb_dim', type=int, default=128)
	parser.add_argument('--seq_hidden_size', type=int, default=128)
	parser.add_argument('--seq_layer_num', type=int, default=1)
	parser.add_argument('--action_method', default='argmax')	# argmax/sample
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

	init_log(args)
	main(args, device)