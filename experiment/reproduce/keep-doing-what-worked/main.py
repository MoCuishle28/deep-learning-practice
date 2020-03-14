import sys
sys.path.append("../..")
import pickle
import argparse
import logging
import datetime
import time
import random

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

from models import Q, Policy, SeqModel, MPO
from historygen import HistoryGenerator


def save_obj(obj, name):
	with open('../../data/ml_1M/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open('../../data/ml_1M/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Algorithm(object):
	def __init__(self, args, agent, device, env):
		super(Algorithm, self).__init__()
		self.args = args
		self.agent = agent
		self.device = device
		self.env = env


	def load_data(self):
		self.train_data = torch.tensor(np.load(self.args.base_data_dir + 'train_data.npy').astype(np.float32), dtype=torch.float32).to(self.device)
		self.train_target = torch.tensor(np.load(self.args.base_data_dir + 'train_target.npy').astype(np.float32), dtype=torch.float32).to(self.device)
		self.valid_data = torch.tensor(np.load(self.args.base_data_dir + 'valid_data.npy').astype(np.float32), dtype=torch.float32).to(self.device)
		self.valid_target = torch.tensor(np.load(self.args.base_data_dir + 'valid_target.npy').astype(np.float32), dtype=torch.float32).to(self.device)

		self.construct_uids_list()
		valid_data_set = Data.TensorDataset(self.valid_data, self.valid_target)
		shuffle = True if args.shuffle == 'y' else False
		self.data_loader = Data.DataLoader(dataset=valid_data_set, batch_size=args.batch_size, shuffle=shuffle)


	def construct_uids_list(self):
		self.train_uids = [i for i in range(1, self.args.max_uid + 1)]


	def sample_uid(self):
		'''
		return: uid, remain
		'''
		max_idx = len(self.train_uids) - 1
		random_idx = random.randint(0, max_idx)
		return self.train_uids.pop(random_idx), max_idx - 1


	def rmse(self, target, prediction):
		rmse = torch.sqrt(torch.sum((prediction - target)**2) / prediction.shape[0])
		return rmse


	def evaluate(self, title='[valid]'):
		self.agent.eval()
		if title == '[test]':
			del self.train_data
			del self.train_target
			del self.valid_data
			del self.valid_target
			test_data = torch.tensor(np.load(self.args.base_data_dir + 'test_data.npy').astype(np.float32), dtype=torch.float32).to(self.device)
			test_target = torch.tensor(np.load(self.args.base_data_dir + 'test_target.npy').astype(np.float32), dtype=torch.float32).to(self.device)
			test_data_set = Data.TensorDataset(test_data, test_target)
			shuffle = True if args.shuffle == 'y' else False
			self.data_loader = Data.DataLoader(dataset=test_data_set, batch_size=args.batch_size, shuffle=shuffle)
		
		sum_rmse = 0
		times = 0
		for i_batch, (data, target) in enumerate(self.data_loader):
			actions = torch.zeros(target.shape[0]).to(self.device)
			for i, feature in enumerate(data):
				uid = feature[0].item()
				mid = feature[1].item()
				s = self.env.get_history(uid, mid).unsqueeze(0).to(self.device)
				a, _, _ = self.agent.select_action_without_noise(s)
				actions[i] = a
			sum_rmse += self.rmse(target, actions)
			times += 1
		print('{} RMSE:{:.6}'.format(title, sum_rmse / times))
		logging.info('{} RMSE:{:.6}'.format(title, sum_rmse / times))


	def train(self):
		for i_epoch in range(self.args.epoch):
			remain = 777
			self.agent.train()
			trajectory_num = 0
			end_idx = 0
			while remain > 0:
				trajectory_num += 1
				if end_idx == 0 or end_idx >= size:
					uid, remain = self.sample_uid()
					behaviors = self.env.get_user_all_behaviors(uid)
					size = len(behaviors)
					end_idx = 0

				curr_mid, curr_rating = behaviors[end_idx]	# rating(action) -> scalar
				end_idx += 1
				curr_s = self.env.get_history(uid, curr_mid)
				self.agent.store_transition(curr_s, curr_rating, 1.0)	# state, action, reward

				for i in range(end_idx, size - 1):
					mid, rating = behaviors[i]
					curr_s = self.env.get_next_history(curr_s, mid, uid, self.agent.action_list[-1])
					self.agent.store_transition(curr_s, rating, 1.0)
					self.agent.store_next_state(curr_s)
					end_idx = i + 1
					if i % self.args.max_len == 0:
						break

				if end_idx < size:
					curr_mid = behaviors[end_idx][0]
					end_idx += 1
					curr_s = self.env.get_next_history(curr_s, curr_mid, uid, self.agent.action_list[-1])
					self.agent.store_next_state(curr_s)
					target = torch.tensor(self.agent.action_list, dtype=torch.float32).to(self.device)
					actions, abm_loss, q_loss, pi_loss, alpha_loss, eta_loss = self.agent.optimize_model()
					rmse = self.rmse(target, actions)

					print_str = 'epoch:{}/{}, trajectory:{}-size:{} RMSE:{:.6}, abm_loss:{:.6}, q_loss:{:.6}, pi_loss:{:.6}, alpha_loss:{:.6}, eta_loss:{:.6}'
					print_str = print_str.format(i_epoch + 1, self.args.epoch, trajectory_num, size, rmse, abm_loss, q_loss, pi_loss, alpha_loss, eta_loss)
					print(print_str)
					logging.info(print_str)
				else:
					self.agent.clear_buffer()

			self.evaluate()
			self.construct_uids_list()

		self.evaluate(title = '[test]')


def init_log(args):
	start = datetime.datetime.now()
	logging.basicConfig(level = logging.INFO,
					filename = args.base_log_dir + args.v + '-' + str(time.time()) + '.log',
					filemode = 'a',		# 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
					)
	logging.info('start! '+str(start))
	logging.info('Parameter:')
	logging.info(str(args))
	logging.info('\n-------------------------------------------------------------\n')


def main(args):
	# init_log(args)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	agent = MPO(args, device)
	env = HistoryGenerator(args, device)
	algorithm = Algorithm(args, agent, device, env)
	algorithm.load_data()
	algorithm.train()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--v', default="v")
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_pic_dir', default="pic/")
	parser.add_argument('--base_data_dir', default='../../data/ml_1M/')

	parser.add_argument('--epoch', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--max_len', type=int, default=256)
	parser.add_argument('--shuffle', default='y')
	parser.add_argument('--act', default='elu')
	parser.add_argument('--optim', default='adam')
	parser.add_argument('--norm_layer', default='ln')		# bn/lb/none
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--max', type=float, default=5.0)
	parser.add_argument('--min', type=float, default=0.0)
	# seq model
	parser.add_argument('--hw', type=int, default=10)
	parser.add_argument('--seq_hidden_size', type=int, default=512)
	parser.add_argument('--seq_layer_num', type=int, default=1)
	parser.add_argument('--seq_output_size', type=int, default=128)
	# embedding
	parser.add_argument('--feature_size', type=int, default=22)	# item feature size(22 dim)
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=128)
	parser.add_argument('--max_mid', type=int, default=193609)	# 1~193609
	parser.add_argument('--m_emb_dim', type=int, default=128)
	parser.add_argument('--g_emb_dim', type=int, default=16)	# genres emb dim
	# RL
	parser.add_argument('--replay_buffer', type=int, default=2*(10**6)) # 2e6
	parser.add_argument('--update_period', type=int, default=200)		# target network update period
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--epsilon', type=float, default=0.1)
	parser.add_argument('--tau', type=float, default=0.1)
	# Q
	parser.add_argument('--q_layers', default='129,256,256')	# state size + action size = 129
	parser.add_argument('--q_lr', type=float, default=1e-3)
	parser.add_argument('--m', type=int, default=20)
	# policy
	parser.add_argument('--a_layers', default='128,256')
	parser.add_argument('--a_lr', type=float, default=1e-4)
	parser.add_argument('--policy_output_size', type=int, default=2)
	# prior
	parser.add_argument('--p_layers', default='128,256')
	parser.add_argument('--p_lr', type=float, default=1e-3)

	args = parser.parse_args()
	main(args)