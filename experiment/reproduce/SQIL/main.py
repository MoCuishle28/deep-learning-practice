import sys
sys.path.append("../..")
import os
import logging
import argparse
import time
import datetime
import random
from collections import deque

import torch
import torch.utils.data as Data
from torch.distributions import Categorical
import numpy as np

from sq import SoftQ, SeqModel
from historygen import HistoryGenerator
from evaluate import Evaluation


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
		self.env = HistoryGenerator(args, device)
		self.demo_replay = deque(maxlen=args.maxlen)
		self.samp_replay = deque(maxlen=args.maxlen)

		self.q = SoftQ(args, device).to(device)

		self.optim = None
		if args.optim == 'adam':
			self.optim = torch.optim.Adam(self.q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		elif args.optim == 'sgd':
			self.optim = torch.optim.SGD(self.q.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif args.optim == 'rmsprop':
			self.optim = torch.optim.RMSprop(self.q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		self.build_data_loader()
		self.evaluate = Evaluation(args, device, self.q, self.env)
		self.EPS = 1e-10


	def fill_replay(self, data):
		# 收集训练数据
		for feature_vector in data.tolist():
			uid, mid = feature_vector[0], feature_vector[1]
			state = self.env.get_history(uid, mid)
			next_state = self.env.get_next_history(state, mid, uid)
			# fill demo_replay
			self.demo_replay.append((state, mid, next_state))
			# fill samp_replay
			samp_action = self.get_action(state)
			# 如果输出的 action 是 expert action 则 state 转移, 否则不转移
			next_state = next_state if samp_action == mid else state
			self.samp_replay.append((state, samp_action, next_state))


	def run(self):
		max_ndcg, max_ndcg_epoch = 0, 0
		hr_list, ndcg_list, precs_list = [], [], []
		for i_epoch in range(self.args.epoch):
			for i_batch, data in enumerate(self.expert_data_loader):
				data = data[0]
				self.fill_replay(data)
				self.train()
				soft_q_loss, demo_error, samp_error = self.update_parameters()
				if i_batch % 10 == 0:
					soft_q_loss, demo_error, samp_error = round(soft_q_loss, 5), round(demo_error, 5), round(samp_error, 5)
					info = f'{i_epoch + 1}/{self.args.epoch}, i_batch:{i_batch}, Soft Q Loss:{soft_q_loss}, Demo Error:{demo_error}, Samp Error:{samp_error}'
					print(info)
					logging.info(info)

			if ((i_epoch + 1) >= self.args.start_eval) and ((i_epoch + 1) % self.args.eval_interval == 0):
				self.eval()
				t1 = time.time()
				hr, ndcg, precs = self.evaluate.eval()
				hr, ndcg, precs = round(hr, 5), round(ndcg, 5), round(precs, 5)
				t2 = time.time()
				max_ndcg = max_ndcg if max_ndcg > ndcg else ndcg
				max_ndcg_epoch = max_ndcg_epoch if max_ndcg > ndcg else i_epoch
				info = f'[Valid]@{self.args.topk} HR:{hr}, NDCG:{ndcg}, Precision:{precs}, Time:{t2 - t1}, Current Max NDCG:{max_ndcg} (epoch:{max_ndcg_epoch})'
				print(info)
				logging.info(info)
				hr_list.append(hr)
				ndcg_list.append(ndcg)
				precs_list.append(precs)

			if ((i_epoch + 1) >= self.args.start_save) and ((i_epoch + 1) % self.args.save_interval == 0):
				self.save_model(version=self.args.v, epoch=i_epoch)
				info = f'Saving version:{args.v}_{i_epoch} models'
				print(info)
				logging.info(info)

		self.eval()
		hr, ndcg, precs = self.evaluate.eval()
		hr, ndcg, precs = round(hr, 5), round(ndcg, 5), round(precs, 5)
		info = f'[TEST]@{self.args.topk} HR:{hr}, NDCG:{ndcg}, Precision:{precs}'
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
		return loss.item(), demo_error.item(), samp_error.item()


	def soft_bellman_error(self, state, action, next_state, reward=1.0):
		'''
		return: (tensor) -> Soft Ballman Error
		'''
		current_q_values = self.q(state)
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


	def build_data_loader(self):
		train_data = torch.tensor(np.load(args.base_data_dir + 'train_data.npy').astype(np.float32), dtype=torch.float32, device=self.device)
		train_data_set = Data.TensorDataset(train_data)
		shuffle = True if self.args.shuffle == 'y' else False
		print('shuffle train data...{}'.format(shuffle))
		self.expert_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=args.batch_size, shuffle=shuffle)


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


	def eval(self):
		self.q.eval()

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
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--v', default="v")
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_pic_dir', default="pic/")
	parser.add_argument('--base_data_dir', default='../../data/ml_1M_row/')
	parser.add_argument('--shuffle', default='y')
	parser.add_argument('--show', default='n')

	parser.add_argument('--load', default='n')			# 是否加载模型
	parser.add_argument('--save', default='y')
	parser.add_argument('--load_version', default='v')
	parser.add_argument('--load_epoch', default='final')
	parser.add_argument('--start_save', type=int, default=50)
	parser.add_argument('--save_interval', type=int, default=20)
	parser.add_argument('--start_eval', type=int, default=0)
	parser.add_argument('--eval_interval', type=int, default=20)

	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=512)

	parser.add_argument('--optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--lr', type=float, default=1e-3)
	# embedding
	parser.add_argument('--feature_size', type=int, default=22)	# uid, mid, genres, ...
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=64)
	parser.add_argument('--max_mid', type=int, default=9741)	# 0~9741
	parser.add_argument('--m_emb_dim', type=int, default=64)
	parser.add_argument('--g_emb_dim', type=int, default=32)	# genres emb dim
	# seq model
	parser.add_argument('--hw', type=int, default=10)
	parser.add_argument('--seq_hidden_size', type=int, default=128)
	parser.add_argument('--seq_layer_num', type=int, default=2)
	parser.add_argument('--seq_output_size', type=int, default=128)
	# Soft Q
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--lammbda_samp', type=float, default=1.0)
	parser.add_argument('--action_method', default='argmax')	# argmax/sample
	parser.add_argument('--maxlen', type=int, default=8000)
	parser.add_argument('--layers', default='128,512,256')		# 第一个对应 seq_output_size
	parser.add_argument('--act', default='elu')
	parser.add_argument('--layer_trick', default='ln')			# ln/bn/none
	parser.add_argument('--dropout', type=float, default=0.0)

	args = parser.parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	init_log(args)
	main(args, device)