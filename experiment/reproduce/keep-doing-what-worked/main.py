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
	with open('../../data/ml_1M_row/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open('../../data/ml_1M_row/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Algorithm(object):
	def __init__(self, args, agent, device, env):
		super(Algorithm, self).__init__()
		self.args = args
		self.agent = agent
		self.device = device
		self.env = env


	def load_data(self):
		train_data = np.load(self.args.base_data_dir + 'seq_predict/' + 'train_data.npy').tolist()
		train_target = np.load(self.args.base_data_dir + 'seq_predict/' + 'train_target.npy').tolist()
		# uid: [mrow, mrow, ...]	下一时刻点击的 item
		self.user_action = {uid:[] for uid in range(1, self.args.max_uid + 1)}

		for pair, prediction in zip(train_data, train_target):
			uid = pair[0]
			self.user_action[uid].append(prediction)

		self.construct_uids_list()


	def construct_uids_list(self):
		self.train_uids = [i for i in range(1, self.args.max_uid + 1)]


	def sample_uid(self):
		'''
		return: uid, remain
		'''
		max_idx = len(self.train_uids) - 1
		random_idx = random.randint(0, max_idx)
		return self.train_uids.pop(random_idx), max_idx - 1


	def train(self):
		for i_epoch in range(self.args.epoch):
			remain = 777
			self.agent.train()
			while remain > 0:
				uid, remain = self.sample_uid()
				for i, action in enumerate(self.user_action[uid]):
					current_state = self.env.get_history(uid, action)
					self.agent.store_transition(current_state, action, 1.0)
					next_state = self.env.get_next_history(current_state, action, uid)
					self.agent.store_next_state(next_state)

					if len(self.agent.reward_list) >= self.args.batch_size:
						precs, abm_loss, q_loss, pi_loss, alpha_loss, eta_loss = self.agent.optimize_model()
						print_str = 'epoch:{}/{}, precs:{:.4}, abm_loss:{:.4}, q_loss:{:.4}, pi_loss:{:.4}, alpha_loss:{:.4}, eta_loss:{:.4}'
						info = print_str.format(i_epoch + 1, self.args.epoch, precs, abm_loss, q_loss, pi_loss, alpha_loss, eta_loss)
						print(info)
						logging.info(info)
				
				if len(self.agent.reward_list) > 0:
					precs, abm_loss, q_loss, pi_loss, alpha_loss, eta_loss = self.agent.optimize_model()
					print_str = 'epoch:{}/{}, precs:{:.4}, abm_loss:{:.4}, q_loss:{:.4}, pi_loss:{:.4}, alpha_loss:{:.4}, eta_loss:{:.4}'
					info = print_str.format(i_epoch + 1, self.args.epoch, precs, abm_loss, q_loss, pi_loss, alpha_loss, eta_loss)
					print(info)
					logging.info(info)
			self.construct_uids_list()
			
			# TODO evaluate valid
			self.agent.eval()
			print('valid')
		# TODO evaluate testing
		print('test')


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
	init_log(args)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	agent = MPO(args, device)
	env = HistoryGenerator(args, device)
	algorithm = Algorithm(args, agent, device, env)
	algorithm.load_data()
	algorithm.train()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--v', default="v")
	parser.add_argument('--num_thread', type=int, default=4)
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_pic_dir', default="pic/")
	parser.add_argument('--base_data_dir', default='../../data/ml_1M_row/')

	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--max_len', type=int, default=256)
	parser.add_argument('--shuffle', default='y')
	parser.add_argument('--act', default='elu')
	parser.add_argument('--optim', default='adam')
	parser.add_argument('--norm_layer', default='ln')		# bn/lb/none
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--max', type=float, default=5.0)
	parser.add_argument('--min', type=float, default=0.0)
	# seq model
	parser.add_argument('--hw', type=int, default=5)
	parser.add_argument('--seq_hidden_size', type=int, default=512)
	parser.add_argument('--seq_layer_num', type=int, default=2)
	parser.add_argument('--seq_output_size', type=int, default=128)
	# embedding
	parser.add_argument('--feature_size', type=int, default=22)	# item feature size(22 dim)
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=128)
	parser.add_argument('--max_mid', type=int, default=9741)	# 0~9741
	parser.add_argument('--m_emb_dim', type=int, default=128)
	parser.add_argument('--g_emb_dim', type=int, default=32)	# genres emb dim
	# RL
	parser.add_argument('--replay_buffer', type=int, default=2*(10**6)) # 2e6
	parser.add_argument('--update_period', type=int, default=200)		# target network update period
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--epsilon', type=float, default=0.1)
	parser.add_argument('--tau', type=float, default=0.1)
	# Q
	parser.add_argument('--q_layers', default='128,256,256')
	parser.add_argument('--q_lr', type=float, default=1e-3)
	parser.add_argument('--m', type=int, default=20)
	# policy
	parser.add_argument('--a_layers', default='128,256')
	parser.add_argument('--a_lr', type=float, default=1e-4)
	# prior
	parser.add_argument('--p_layers', default='128,256')
	parser.add_argument('--p_lr', type=float, default=1e-3)

	args = parser.parse_args()
	main(args)