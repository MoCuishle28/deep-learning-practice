import pickle
import argparse
import random
import logging
import datetime
import time
from collections import deque

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np

from seqddpg import DDPG
from seqddpg import Transition
from seqddpg import ReplayMemory
from seqddpg import OUNoise
from myModel import FM, NCF
from myModel import Predictor
from evaluate import Evaluate


def save_obj(obj, name):
	with open('../data/ml_1M_row/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open('../data/ml_1M_row/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Algorithm(object):
	def __init__(self, args, agent, predictor, env, data_list, device):
		self.device = device
		self.args = args
		self.agent = agent
		self.ounoise = OUNoise(args.actor_output)
		self.predictor = predictor
		self.memory = ReplayMemory(args.memory_size)
		self.env = env

		self.train_data = torch.tensor(data_list.pop(0), dtype=torch.float32, device=self.device)
		self.valid_data = torch.tensor(data_list.pop(0), dtype=torch.float32, device=self.device)
		self.test_data = torch.tensor(data_list.pop(0), dtype=torch.float32, device=self.device)

		train_data_set = Data.TensorDataset(self.train_data)
		shuffle = True if args.shuffle == 'y' else False
		print('shuffle train data...{}'.format(shuffle))
		self.train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=args.batch_size, shuffle=shuffle)

		self.evaluate = Evaluate(args, device, agent, predictor, self.train_data, self.valid_data, self.test_data, env)


	def collect_training_data(self, data):
		self.agent.on_eval()	# 切换为评估模式
		self.predictor.on_train()

		state_list = []
		action_list = []
		next_state_list = []
		input_data_list = []

		for i_data, raw_feature in enumerate(data):
			state = self.env.get_history(raw_feature[0].item(), raw_feature[1].item())
			next_state = self.env.get_next_history(state, raw_feature[1].item(), raw_feature[0].item())

			# 转成符合 RNN 的输入数据形式
			state = state.view(-1, state.shape[0], state.shape[1]).to(self.device)
			next_state = next_state.view(-1, next_state.shape[0], next_state.shape[1]).to(self.device)
			action = self.agent.select_action(state, action_noise=self.ounoise)

			state_list.append(state)
			action_list.append(action)
			next_state_list.append(next_state)

			input_data = torch.cat([action.squeeze(), raw_feature])		# action (1, actor_output) -> (actor_output)
			input_data = input_data.view(1, -1)	# input_data (actor_output + 22) -> (1, actor_output + 22)
			input_data_list.append(input_data)
		if self.args.reset == 'y':
			self.ounoise.reset()	# reset noise vector

		batch_input_data = torch.cat(input_data_list, dim=0).to(self.device)
		# 训练 predictor
		sum_bpr_loss, batch_bpr_loss = self.predictor.train(batch_input_data)

		# bpr loss 的负数作为 reward
		if self.args.reward == 'loss':
			bpr_loss_list = batch_bpr_loss.tolist()
			for state, action, next_state, bpr_loss in zip(state_list, action_list, next_state_list, bpr_loss_list):
				reward = torch.tensor([bpr_loss[0] * self.args.alpha], dtype=torch.float32, device=self.device)
				mask = torch.tensor([True], dtype=torch.float32, device=self.device)
				self.memory.push(state, action, mask, next_state, reward)
		return sum_bpr_loss, batch_bpr_loss


	def train(self):
		average_reward_list = []
		bpr_loss_list = []
		precision_list = []
		hr_list = []
		ndcg_list = []
		for epoch in range(self.args.epoch):
			for i_batch, data in enumerate(self.train_data_loader):
				data = data[0]				
				sum_bpr_loss, batch_bpr_loss = self.collect_training_data(data)

				transitions = self.memory.sample(self.args.batch_size)
				batch = Transition(*zip(*transitions))
				self.agent.on_train()	# 切换为训练模式 (因为包含 BN\LN)
				value_loss, policy_loss = self.agent.update_parameters(batch)

				predictor_loss_mean = batch_bpr_loss.mean().item()
				reward = 0
				if self.args.reward == 'loss':
					# Average Reward e.g. Negative Average predictor loss
					reward = predictor_loss_mean * self.args.alpha

				average_reward_list.append(reward)
				bpr_loss_list.append(sum_bpr_loss.item())

				print('epoch:{}/{} @{} i_batch:{}, Average Reward:{:.4}, BPR LOSS:{:.4}'.format(epoch+1, self.args.epoch, 
					self.args.topk, i_batch+1, reward, sum_bpr_loss.item()), end = ', ')
				print('value loss:{:.4}, policy loss:{:.4}'.format(value_loss, policy_loss))
				logging.info('epoch:{}/{} @{} i_batch:{}, Average Reward:{:.4}'.format(epoch+1, self.args.epoch, 
					self.args.topk, i_batch+1, reward))

			t1 = time.time()
			hr, ndcg, precs = self.evaluate.evaluate()
			t2 = time.time()
			print('[Valid]@{} HR:{:.4}, NDCG:{:.4}, Precision:{:.4}, Time:{}'.format(self.args.topk, hr, ndcg, precs, t2 - t1))
			logging.info('[Valid]@{} HR:{:.4}, NDCG:{:.4}, Precision:{:.4}, Time:{}'.format(self.args.topk, hr, ndcg, precs, t2 - t1))
			hr_list.append(hr)
			ndcg_list.append(ndcg)
			precision_list.append(precs)

		hr, ndcg, precs = self.evaluate.evaluate(title='[TEST]')
		print('[TEST]@{} HR:{:.4}, NDCG:{:.4}, Precision:{:.4}'.format(self.args.topk, hr, ndcg, precs))
		logging.info('[TEST]@{} HR:{:.4}, NDCG:{:.4}, Precision:{:.4}'.format(self.args.topk, hr, ndcg, precs))
		self.evaluate.plot_result(args, bpr_loss_list, precision_list, hr_list, ndcg_list)


class HistoryGenerator(object):
	def __init__(self, args, device):
		self.device = device
		self.args = args
		# mid: one-hot feature (21维 -> mid, genre, genre, ...)
		self.mid_map_mfeature = load_obj('mid_map_mfeature')	
		self.users_rating = load_obj('users_rating_without_timestamp') # uid:[[mid, rating], ...] 有序
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
				mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32).to(self.device)
				# [uid, mfeature...]
				history_feature = torch.cat([torch.tensor([uid], dtype=torch.float32).to(self.device), 
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
		mfeature = torch.tensor(self.mid_map_mfeature[new_mid].astype(np.float32), dtype=torch.float32).to(self.device)		

		history_feature = torch.cat([torch.tensor([uid], dtype=torch.float32, device=self.device), mfeature]).to(self.device)
		curr_history.append(history_feature)
		return torch.tensor(curr_history).to(self.device)


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
	train_data = np.load(args.base_data_dir + 'train_data.npy').astype(np.float32)
	valid_data = np.load(args.base_data_dir + 'valid_data.npy').astype(np.float32)
	test_data = np.load(args.base_data_dir + 'test_data.npy').astype(np.float32)
	data_list = [train_data] + [valid_data] + [test_data]

	env = HistoryGenerator(args, device)
	agent = DDPG(args, device)

	# 后面还可以改成他的预测 rating 算法
	predictor_model = None
	if args.predictor == 'fm':
		predictor_model = FM(args.u_emb_dim + args.m_emb_dim + args.g_emb_dim + args.actor_output, args.k, args, device)
		print('predictor_model is FM.')
		logging.info('predictor_model is FM.')
	elif args.predictor ==  'ncf':
		predictor_model = NCF(args, args.u_emb_dim + args.actor_output + args.m_emb_dim + args.g_emb_dim, device)
		print('predictor_model is NCF.')
		logging.info('predictor_model is NCF.')

	predictor = Predictor(args, predictor_model, device, env.mid_map_mfeature)

	# 加载模型
	if args.load == 'y':
		agent.load_model(version=args.v)
		predictor.load(args.v)
		print('Loading version:{} models'.format(args.v))
		logging.info('Loading version:{} models'.format(args.v))

	algorithm = Algorithm(args, agent, predictor, env, data_list, device)
	algorithm.train()

	# 保存模型
	if args.save == 'y':
		algorithm.agent.save_model(version=args.v)
		predictor.save(args.v)
		print('Saving version:{} models'.format(args.v))
		logging.info('Saving version:{} models'.format(args.v))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters for DDPG and Predictor")
	parser.add_argument('--v', default="v")
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--num_thread', type=int, default=4)	# 用 GPU 跑时设为 0

	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_pic_dir', default="pic/")
	parser.add_argument('--base_data_dir', default='../data/ml_1M_row/')

	parser.add_argument('--epoch', type=int, default=5)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--shuffle', default='y')
	# RL setting
	parser.add_argument('--reset', default='y')
	parser.add_argument('--memory_size', type=int, default=10000)
	parser.add_argument('--hw', type=int, default=10)	# history window
	parser.add_argument('--predictor', default='fm')	# fm/ncf
	parser.add_argument('--reward', default='loss')
	parser.add_argument('--alpha', type=float, default=1)	# raw reward 乘以的倍数(试图放大 reward 加大训练幅度)

	parser.add_argument('--predictor_optim', default='adam')
	parser.add_argument('--actor_optim', default='adam')
	parser.add_argument('--critic_optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)	# sgd 时
	parser.add_argument('--weight_decay', type=float, default=1e-4)		# regularization
	parser.add_argument('--norm_layer', default='ln')			# bn/ln/none
	parser.add_argument('--dropout', type=float, default=0.0)	# dropout (BN 可以不需要)
	# save/load model 的名字为 --v
	parser.add_argument('--save', default='n')
	parser.add_argument('--load', default='n')
	parser.add_argument('--show_pic', default='n')
	# init weight
	parser.add_argument('--init_std', type=float, default=0.1)
	# seq model
	parser.add_argument('--seq_hidden_size', type=int, default=512)
	parser.add_argument('--seq_layer_num', type=int, default=2)
	parser.add_argument('--seq_output_size', type=int, default=128)
	# ddpg
	parser.add_argument("--actor_lr", type=float, default=1e-5)
	parser.add_argument("--critic_lr", type=float, default=1e-4)
	parser.add_argument('--hidden_size', type=int, default=512)
	parser.add_argument('--actor_output', type=int, default=64)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--actor_tau', type=float, default=0.1)
	parser.add_argument('--critic_tau', type=float, default=0.1)
	parser.add_argument('--a_act', default='elu')
	parser.add_argument('--c_act', default='elu')
	# predictor
	parser.add_argument("--predictor_lr", type=float, default=1e-4)
	# embedding
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=64)
	parser.add_argument('--max_mid', type=int, default=9741)	# 0~9741
	parser.add_argument('--m_emb_dim', type=int, default=128)
	parser.add_argument('--g_emb_dim', type=int, default=32)	# genres emb dim
	# FM
	parser.add_argument('--fm_feature_size', type=int, default=22)	# 还要原来基础加上 actor_output
	parser.add_argument('--k', type=int, default=8)
	# NCF
	parser.add_argument('--n_act', default='elu')
	parser.add_argument('--layers', default='1024,512')

	args = parser.parse_args()
	init_log(args)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('device:{}'.format(device))
	main(args, device)