import sys
sys.path.append("..")
import os
import argparse
import logging
import multiprocessing
import time
import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from seqddpg import DDPG
from myModel import FM, NCF
from myModel import Predictor
from evaluate import Evaluate
from utils import Utils


class Run(object):
	def __init__(self, args, device, predictor, agent, env):
		super(Run, self).__init__()
		self.args = args
		self.device = device
		self.predictor = predictor
		self.agent = agent

		self.valid_data = np.load(args.base_data_dir + 'valid_data.npy').astype(np.float32)
		self.test_data = np.load(args.base_data_dir + 'test_data.npy').astype(np.float32)
		self.evaluate = Evaluate(args, device, agent, predictor, None, self.valid_data, self.test_data, env)


	def eval_RL(self):
		hr_list, ndcg_list, precision_list = [], [], []
		if not os.path.exists(f'models/{self.args.v}/'):
			print(f'version{self.args.v} not exists!')
			return
		file_names = os.listdir(f"./models/{self.args.v}/")
		version_list = list(set([x.split('-')[-1].split('.')[0] for x in file_names]))

		for epoch in version_list:
			# 加载模型
			self.agent.load_model(version=self.args.v, epoch=epoch)
			self.predictor.load(self.args.v, epoch=epoch)
			self.agent.on_eval()
			self.predictor.on_eval()
			info = f'Loading version:{self.args.v}_{epoch} models'
			print(info)
			logging.info(info)

			self.evaluate.predictor = self.predictor
			self.evaluate.agent = self.agent
			t1 = time.time()
			hr, ndcg, precs = self.evaluate.evaluate()
			hr, ndcg, precs = round(hr, 5), round(ndcg, 5), round(precs, 5)
			t2 = time.time()
			info = f'[Valid]@{self.args.topk} | epoch:{epoch} HR:{hr}, NDCG:{ndcg}, Precision:{precs}, Time:{t2 - t1}'
			print(info)
			logging.info(info)
			hr_list.append(hr)
			ndcg_list.append(ndcg)
			precision_list.append(precs)


	def eval_only_predictor(self):
		pass


class HistoryGenerator(object):
	def __init__(self, args, device):
		self.device = device
		self.args = args
		# mid: one-hot feature (21维 -> mid, genre, genre, ...)
		self.mid_map_mfeature = Utils.load_obj(args.base_data_dir, 'mid_map_mfeature')
		# uid:[[mid, rating], ...] 有序
		self.users_rating = Utils.load_obj(args.base_data_dir, 'users_rating_without_timestamp')
		self.window = args.hw
		self.build_index()


	def build_index(self):
		'''建立 uid, mid 的索引'''
		self.index = {}		# uid: {mid: idx, ...}, ...
		for uid, items_list in self.users_rating.items():
			self.index[uid] = {}
			for i, item in enumerate(items_list):
				self.index[uid][item[0]] = i


	def get_history(self, uid, curr_mid):
		'''
		return: tensor (window, feature size:23 dim -> [uid, mid, genre, rating])
		'''
		ret_data = []
		rating_list = self.users_rating[uid]
		stop_index = self.index[uid][curr_mid]
		for i in range(stop_index - self.window, stop_index):
			if i < 0:
				history_feature = torch.zeros(22, dtype=torch.float32, device=self.device)
				history_feature[0] = uid
				history_feature[1] = 9742.0
			else:
				mid = rating_list[i][0]
				mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32, device=self.device)
				# [uid, mfeature...]
				history_feature = torch.cat([torch.tensor([uid], dtype=torch.float32, device=self.device), 
					mfeature]).to(self.device)

			ret_data.append(history_feature)
		return torch.stack(ret_data).to(self.device)


	def get_next_history(self, curr_history, new_mid, curr_uid):
		'''
		这个 state 的转移方式没有考虑 action(即: trasition probability = 1)
		curr_history: tensor (window, feature size)
		return: tensor (window, feature size)
		'''
		curr_history = curr_history.tolist()
		curr_history.pop(0)
		uid = curr_uid
		mfeature = torch.tensor(self.mid_map_mfeature[new_mid].astype(np.float32), dtype=torch.float32, device=self.device)

		history_feature = torch.cat([torch.tensor([uid], dtype=torch.float32, device=self.device), mfeature]).to(self.device)
		curr_history.append(history_feature)
		return torch.tensor(curr_history, device=self.device)


def main(args, device):
	env = None
	agent = None
	predictor_model = None
	predictor = None

	info = f'Predictor is {args.predictor}.'
	print(info)
	logging.info(info)
	if args.choose == 'rl':		# 评估 RL + Predictor
		env = HistoryGenerator(args, device)
		agent = DDPG(args, device)
		if args.predictor == 'fm':
			predictor_model = FM(args.u_emb_dim + args.m_emb_dim + args.g_emb_dim + args.actor_output, args.k, args, device)
		elif args.predictor ==  'ncf':
			predictor_model = NCF(args, args.u_emb_dim + args.actor_output + args.m_emb_dim + args.g_emb_dim, device)
		predictor = Predictor(args, predictor_model, device, env.mid_map_mfeature)
	else:						# 评估 Predictor
		if args.predictor == 'fm':
			predictor_model = FM(args.u_emb_dim + args.m_emb_dim + args.g_emb_dim, args.k, args, device, without_rl=True)
		elif args.predictor == 'ncf':
			predictor_model = NCF(args, args.u_emb_dim + args.m_emb_dim + args.g_emb_dim, device, without_rl=True)
		mid_map_mfeature = Utils.load_obj(args.base_data_dir, 'mid_map_mfeature')
		predictor = Predictor(args, predictor_model, device, mid_map_mfeature)
		
	run = Run(args, device, predictor, agent, env)
	if args.choose == 'rl':
		run.eval_RL()
	else:
		run.eval_only_predictor()


def init_log(args, device):
	if not os.path.exists(args.base_log_dir):
		os.makedirs(args.base_log_dir)

	start = datetime.datetime.now()
	logging.basicConfig(level = logging.INFO,
					filename = args.base_log_dir + args.v + '-' + str(time.time()) + '.log',
					filemode = 'a',		# 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
					# a是追加模式，默认如果不写的话，就是追加模式
					)
	logging.info('start! '+str(start))
	logging.info(f'device:{device}')
	logging.info('Parameter:')
	logging.info(str(args))
	logging.info('\n-------------------------------------------------------------\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters for Off Line Evaluation")
	parser.add_argument('--v', default="v")
	parser.add_argument('--choose', default="rl")				# 评估 RL+Predictor 还是 predictor
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--num_thread', type=int, default=4)	# 用 GPU 跑时设为 0

	parser.add_argument('--base_log_dir', default="eval/")
	parser.add_argument('--base_data_dir', default='../data/ml_1M_row/')
	parser.add_argument('--hw', type=int, default=10)	# history window
	parser.add_argument('--predictor', default='fm')	# fm/ncf
	# loss/posscore/dismargin(离散)/conmargin(连续)
	parser.add_argument('--reward', default='dismargin')
	parser.add_argument('--alpha', type=float, default=1)
	# seq model
	parser.add_argument('--seq_hidden_size', type=int, default=512)
	parser.add_argument('--seq_layer_num', type=int, default=2)
	parser.add_argument('--seq_output_size', type=int, default=128)
	# ddpg
	parser.add_argument('--norm_layer', default='ln')			# bn/ln/none
	parser.add_argument('--hidden_size', type=int, default=512)
	parser.add_argument('--actor_output', type=int, default=64)
	parser.add_argument('--a_act', default='elu')
	parser.add_argument('--c_act', default='elu')
	# embedding
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=64)
	parser.add_argument('--max_mid', type=int, default=9741)	# 0~9741
	parser.add_argument('--m_emb_dim', type=int, default=128)
	parser.add_argument('--g_emb_dim', type=int, default=32)	# genres emb dim
	# FM
	parser.add_argument('--fm_feature_size', type=int, default=22)	# 还要原来基础加上 actor_output
	parser.add_argument('--k', type=int, default=64)
	# NCF
	parser.add_argument('--n_act', default='elu')
	parser.add_argument('--layers', default='1024,512')

	# 占位
	parser.add_argument("--actor_lr", type=float, default=0)
	parser.add_argument("--critic_lr", type=float, default=0)
	parser.add_argument("--predictor_lr", type=float, default=0)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--actor_tau', type=float, default=0.1)
	parser.add_argument('--critic_tau', type=float, default=0.1)
	parser.add_argument('--predictor_optim', default='adam')
	parser.add_argument('--actor_optim', default='adam')
	parser.add_argument('--critic_optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--init_std', type=float, default=0.1)

	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tmp = 'cuda' if torch.cuda.is_available() else 'cpu'
	args.num_thread = 0 if tmp == 'cuda' else args.num_thread

	print(f'device:{device}')
	init_log(args, device)
	main(args, device)