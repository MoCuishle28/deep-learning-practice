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

		plt.savefig(self.args.base_pic_dir + self.args.v + '.png')
		if self.args.show == 'y':
			plt.show()


class SoftQ(nn.Module):
	def __init__(self, args, device):
		super(SoftQ, self).__init__()
		self.args = args
		self.device = device
		self.seq_input_size = args.m_emb_dim
		self.hidden_size = args.seq_hidden_size
		self.seq_layer_num = args.seq_layer_num

		self.m_embedding = nn.Embedding(args.max_iid + 1 + 1, args.m_emb_dim)	# 初始状态 mid=70852

		dropout = args.dropout if self.seq_layer_num > 1 else 0.0
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.gru = nn.GRU(self.seq_input_size, self.hidden_size, self.seq_layer_num, batch_first=True, dropout=dropout)
		self.ln = None
		if args.layer_trick == 'bn':
			self.ln = nn.BatchNorm1d(self.hidden_size, affine=True)
		elif args.layer_trick == 'ln':
			self.ln = nn.LayerNorm(self.hidden_size, elementwise_affine=True)		
		self.fc = nn.Linear(self.hidden_size, args.max_iid + 1)

	
	def forward(self, x):
		x = self.m_embedding(x.long().to(self.device))
		x = x.view(x.shape[0], x.shape[1], -1)
		# 需要 requires_grad=True 吗？
		h0 = torch.zeros(self.seq_layer_num, x.size(0), self.hidden_size, device=self.device)

		out, _ = self.gru(x, h0)  	# out: tensor of shape (batch_size, seq_length, hidden_size)
		out = out[:, -1, :]			# 最后时刻的 seq 作为输出
		if self.args.layer_trick != 'none':
			out = self.ln(out)
		out = self.fc(out)
		return out


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
		self.q = SoftQ(args, device).to(device)
		if args.target == 'y':
			self.target_q = SoftQ(args, device).to(device)
			hard_update(self.target_q, self.q)

		self.optim = None
		if args.optim == 'sgd':
			self.optim = torch.optim.SGD(self.q.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif args.optim == 'rmsprop':
			self.optim = torch.optim.RMSprop(self.q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		else:
			self.optim = torch.optim.Adam(self.q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		# 每次调用 self.scheduler.step(),都降低 lr
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, args.lr_decay)

		self.evaluate = Evaluation(args, device, self.q)
		self.EPS = 1e-10
		self.build_data_loder()
		self.samp_replay = deque(maxlen=args.maxlen)


	def build_data_loder(self):
		state_list = torch.tensor(np.load(self.args.base_data_dir + 'state.npy'), dtype=torch.float32, device=self.device)
		next_state_list = torch.tensor(np.load(self.args.base_data_dir + 'next_state.npy'), dtype=torch.float32, device=self.device)
		action_list = torch.tensor(np.load(self.args.base_data_dir + 'action.npy'), device=self.device)
		self.args.maxlen = state_list.shape[0]
		self.demo_replay = deque(maxlen=self.args.maxlen)
		self.fill_expert_replay(state_list, next_state_list, action_list)

		dataset = Data.TensorDataset(state_list, next_state_list, action_list)
		shuffle = True if self.args.shuffle == 'y' else False
		print('shuffle train data...{}'.format(shuffle))
		self.data_loader = Data.DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=shuffle)


	def fill_expert_replay(self, state_list, next_state_list, action_list):
		print(f'fill expert replay, maxlen:{self.args.maxlen}...')
		for state, next_state, action in zip(state_list, next_state_list, action_list):
			# fill demo_replay
			self.demo_replay.append((state, action, next_state))


	def fill_replay(self, data):
		# 收集训练数据
		for state, next_state, action in zip(data[0].tolist(), data[1].tolist(), data[2].tolist()):
			state = torch.tensor(state, dtype=torch.float32, device=self.device).view(-1, 1)
			next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).view(-1, 1)

			samp_action = self.get_action(state)
			# 如果输出的 action 是 expert action 则 state 转移, 否则不转移
			next_state = next_state if samp_action == action else state
			self.samp_replay.append((state, samp_action, next_state))


	def run(self):
		max_ndcg, max_ndcg_epoch = 0, 0
		hr_list, ndcg_list, precs_list = [], [], []
		no_improve_times = 0
		for i_epoch in range(self.args.epoch):
			for i_batch, data in enumerate(self.data_loader):
				self.fill_replay(data)
				self.train()
				soft_q_loss, demo_error, samp_error = self.update_parameters()
				if i_batch % 200 == 0:
					soft_q_loss, demo_error, samp_error = round(soft_q_loss, 5), round(demo_error, 5), round(samp_error, 5)
					info = f'{i_epoch + 1}/{self.args.epoch}, i_batch:{i_batch}, Soft Q Loss:{soft_q_loss}, Demo Error:{demo_error}, Samp Error:{samp_error}'
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


	def update_parameters(self):
		# demo replay
		state, action, next_state = self.sample_data(self.demo_replay)
		demo_error = self.soft_bellman_error(state, action, next_state)
		# samp replay
		state, action, next_state = self.sample_data(self.samp_replay)
		samp_error = self.soft_bellman_error(state, action, next_state, reward=0.0)

		loss = demo_error + self.args.lammbda_samp * samp_error
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		if self.args.target == 'y':
			soft_update(self.target_q, self.q, self.args.tau)
		return loss.item(), demo_error.item(), samp_error.item()


	def soft_bellman_error(self, state, action, next_state, reward=1.0):
		'''
		return: (tensor) -> Soft Ballman Error
		'''
		current_q_values = self.q(state)
		if self.args.target == 'y':
			self.target_q.eval()
			with torch.no_grad():
				next_q_values = self.target_q(next_state)
				next_q_values.detach_()
		else:
			next_q_values = self.q(next_state)

		action = action.view(-1, 1)		# (batch, 1), dtype=int64
		current_q_values = torch.gather(current_q_values, 1, action).squeeze()		# (batch)
		exp_next_q_values = torch.sum(torch.exp(next_q_values), dim=-1)				# (batch)
		error = (current_q_values - (reward + self.args.gamma * torch.log(exp_next_q_values + self.EPS)))**2
		error = error.mean()
		return error


	def sample_data(self, replay):
		'''
		return: (tensor) -> state, action, next_state
		'''
		batch_state, batch_action, batch_next_state = zip(
			*random.sample(replay, self.args.batch_size))
		batch_action = torch.tensor(batch_action, dtype=torch.long, device=self.device)	# (512), int64
		batch_state = torch.stack(batch_state).to(self.device)				# (512, 10, 22)
		batch_next_state = torch.stack(batch_next_state).to(self.device)	# (512, 10, 22)
		return batch_state, batch_action, batch_next_state


	def get_action(self, state):
		state = state.view(1, state.shape[0], state.shape[1])
		q_values = self.q(state)

		if self.args.action_method == 'argmax':
			action = q_values.argmax().item()	# mid
		else:
			dist = Categorical(torch.tensor(torch.softmax(q_values, dim=-1), device=self.device))
			action = dist.sample().item()
		return action

	def save_model(self, version, epoch):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		if not os.path.exists('models/' + version + '/'):
			os.makedirs('models/' + version + '/')

		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		torch.save(self.q.state_dict(), based_dir + 'softQ_' + tail)

	def load_model(self, version, epoch):
		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		self.q.load_state_dict(torch.load(based_dir + 'softQ_' + tail))
		if self.args.target == 'y':
			hard_update(self.target_q, self.q)

	def eval(self):
		self.q.eval()
		if self.args.target == 'y':
			self.target_q.eval()

	def train(self):
		self.q.train()


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
	# Soft Q
	parser.add_argument('--seq_hidden_size', type=int, default=128)
	parser.add_argument('--seq_layer_num', type=int, default=1)
	parser.add_argument('--tau', type=float, default=0.9)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--lammbda_samp', type=float, default=1.0)
	parser.add_argument('--action_method', default='argmax')	# argmax/sample
	parser.add_argument('--maxlen', type=int, default=80000)
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