import os
import pickle
import argparse
import random
import logging
import datetime
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from seqddpg import DDPG
from seqsac import SAC
from myfm import Predictor
from myfm import FM, Net, NCF

'''
SAC
offline-eval.py --v v28 --load_epoch 39 --agent sac
offline-eval.py --v v29 --load_epoch 29 --agent sac --predictor ncf

DDPG
offline-eval.py --v v30 --load_epoch 29
offline-eval.py --v v27 --load_epoch 29 --predictor ncf
'''


def save_obj(obj, name):
	with open('../../data/ml_1M_row/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open('../../data/ml_1M_row/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Evaluation(object):
	def __init__(self, args, device, agent, predictor, env):
		super(Evaluation, self).__init__()
		self.args = args
		self.device = device
		self.agent = agent
		self.predictor = predictor
		self.env = env
		self.test_data = torch.tensor(np.load(args.base_data_dir + 'test_data.npy').astype(np.float32), dtype=torch.float32, device=device)
		self.test_target = torch.tensor(np.load(args.base_data_dir + 'test_target.npy').astype(np.float32), dtype=torch.float32, device=device)


	def eval(self):
		self.agent.on_eval()
		self.predictor.on_eval()
		with torch.no_grad():
			input_data = []
			for i_data, raw_feature in enumerate(self.test_data):
				state = self.env.get_history(raw_feature[0].item(), raw_feature[1].item())
				state = state.reshape((-1, state.shape[0], state.shape[1])).to(self.device)

				action = self.agent.select_action(state, action_noise=None)	# 不加噪声
				input_data.append(torch.cat([action.squeeze(), raw_feature]))

			input_data = torch.stack(input_data).to(self.device)
			prediction, _ = self.predictor.predict(input_data, self.test_target)
			rmse = self.get_rmse(prediction, self.test_target)
			mae = self.get_mae(prediction, self.test_target)

			rmse, mae = round(rmse, 5), round(mae, 5)
			info = f'RMSE:{rmse}, MAE:{mae}'
			print(info)
			logging.info(info)


	def get_rmse(self, prediction, target):
		if prediction.shape != target.shape:
			prediction = prediction.squeeze()
		rmse = torch.sqrt(((prediction - target)**2).mean())
		return rmse.item()


	def get_mae(self, prediction, target):
		if prediction.shape != target.shape:
			prediction = prediction.squeeze()
		mae = (torch.abs(prediction - target)).mean()
		return mae.item()


class HistoryGenerator(object):
	def __init__(self, args, device):
		self.device = device
		self.args = args
		# mid: one-hot feature (21维 -> mid, genre, genre, ...)
		self.mid_map_mfeature = load_obj('mid_map_mfeature')		
		self.users_rating = load_obj('users_rating_without_timestamp') # uid:[[mid, rating], ...] 有序
		# self.users_has_clicked = load_obj('users_has_clicked')	# uid:{mid, mid, ...}
		self.window = args.hw
		self.compute_mean_std()
		self.build_index()


	def compute_mean_std(self):
		rating_list = []
		for uid, behavior_list in self.users_rating.items():
			for pair in behavior_list:
				rating_list.append(pair[-1])

		rating_tensor = torch.tensor(rating_list, dtype=torch.float32).to(self.device)
		self.rating_mean, self.rating_std = rating_tensor.mean(), rating_tensor.std()


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
				history_feature = torch.zeros(23, dtype=torch.float32, device=self.device)
				history_feature[0] = uid
				history_feature[1] = self.args.max_mid
			else:
				mid = rating_list[i][0]
				rating  = rating_list[i][1]
				mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32, device=self.device)
				# [uid, mfeature..., rating]
				history_feature = torch.cat([torch.tensor([uid], dtype=torch.float32).to(self.device), 
					mfeature, 
					torch.tensor([rating], dtype=torch.float32).to(self.device)]).to(self.device)

			history_feature[-1] = (history_feature[-1] - self.rating_mean) / self.rating_std
			ret_data.append(history_feature)
		return torch.stack(ret_data).to(self.device)


	def get_next_history(self, curr_history, new_mid, curr_uid, rating):
		'''
		这个 state 的转移方式没有考虑 action(即: trasition probability = 1)
		curr_history: tensor (window, feature size)
		return: tensor (window, feature size)
		'''
		curr_history = curr_history.tolist()
		curr_history.pop(0)
		uid = curr_uid
		mfeature = torch.tensor(self.mid_map_mfeature[new_mid].astype(np.float32), dtype=torch.float32, device=self.device)
		rating = torch.tensor([rating], dtype=torch.float32).to(self.device)
		
		history_feature = torch.cat([torch.tensor([uid], dtype=torch.float32).to(self.device), mfeature, rating]).to(self.device)
		history_feature[-1] = (history_feature[-1] - self.rating_mean) / self.rating_std
		curr_history.append(history_feature)
		return torch.tensor(curr_history, device=self.device)


def main(args, device):
	env = HistoryGenerator(args, device)

	print(f'agent is {args.agent}')
	if args.agent == 'ddpg':
		agent = DDPG(args, device)
	elif args.agent == 'sac':
		agent = SAC(args, device)

	predictor_model = None
	if args.predictor == 'fm':
		predictor_model = FM(args.u_emb_dim + args.m_emb_dim + args.g_emb_dim + args.actor_output, args.k, args, device)
		print('predictor_model is FM.')
		logging.info('predictor_model is FM.')
	elif args.predictor ==  'ncf':
		predictor_model = NCF(args, args.u_emb_dim + args.actor_output + args.m_emb_dim + args.g_emb_dim, device)
		print('predictor_model is NCF.')
		logging.info('predictor_model is NCF.')
	predictor = Predictor(args, predictor_model, device)

	# 加载模型	
	agent.load_model(version=args.v, epoch=args.load_epoch)
	predictor.load(args.v, epoch=args.load_epoch)
	info = f'Loading version:{args.v}_{args.load_epoch} models'
	print(info)
	logging.info(info)

	evaluate = Evaluation(args, device, agent, predictor, env)
	evaluate.eval()


def init_log(args, device):
	if not os.path.exists(args.base_log_dir):
		os.makedirs(args.base_log_dir)
	if not os.path.exists(args.base_pic_dir):
		os.makedirs(args.base_pic_dir)

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
	parser.add_argument('--base_log_dir', default="eval/log/")
	parser.add_argument('--base_pic_dir', default="eval/pic/")
	parser.add_argument('--base_data_dir', default='../../data/ml_1M_row/')
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--rl', default='y')		# yes/no
	parser.add_argument('--device', default='cpu')		# yes/no

	parser.add_argument('--agent', default='ddpg')
	parser.add_argument('--epoch', type=int, default=5)
	parser.add_argument('--hw', type=int, default=10)	# history window
	parser.add_argument('--predictor', default='fm')
	# rating 范围
	parser.add_argument('--min', type=float, default=0.0)
	parser.add_argument('--max', type=float, default=5.0)

	# model 的名字为 --v
	parser.add_argument('--v', default="v")
	parser.add_argument('--load_epoch', default='final')
	parser.add_argument('--show_pic', default='n')

	parser.add_argument('--norm_layer', default='ln')			# bn/ln/none
	# seq model
	parser.add_argument('--seq_hidden_size', type=int, default=256)
	parser.add_argument('--seq_layer_num', type=int, default=2)
	parser.add_argument('--seq_output_size', type=int, default=128)
	# ddpg
	parser.add_argument('--hidden_size', type=int, default=512)
	parser.add_argument('--actor_output', type=int, default=64)
	parser.add_argument('--a_act', default='elu')
	parser.add_argument('--c_act', default='elu')
	# sac
	parser.add_argument('--v_act', default='elu')
	parser.add_argument('--log_std_min', type=float, default=-20)
	parser.add_argument('--log_std_max', type=float, default=2)
	# predictor
	parser.add_argument('--n_act', default='elu')
	# embedding
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=64)
	parser.add_argument('--max_mid', type=int, default=9742)	# 0~9741 + 1
	parser.add_argument('--m_emb_dim', type=int, default=128)
	parser.add_argument('--g_emb_dim', type=int, default=16)	# genres emb dim
	# FM
	parser.add_argument('--fm_feature_size', type=int, default=22)	# 还要原来基础加上 actor_output
	parser.add_argument('--k', type=int, default=8)
	# NCF
	parser.add_argument('--layers', default='1024,512,256')
	# 占位
	parser.add_argument('--actor_optim', default='none')
	parser.add_argument('--critic_optim', default='none')
	parser.add_argument('--predictor_optim', default='none')
	parser.add_argument('--dropout', type=float, default=0.0)	# eval 下不需要
	parser.add_argument('--init_std', type=float, default=0.1)

	args = parser.parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print('device:{}'.format(device))

	# 保持可复现
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)
	init_log(args, device)
	main(args, device)