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
import matplotlib.pyplot as plt

from seqddpg import DDPG
from seqddpg import Transition
from seqddpg import ReplayMemory
from seqddpg import OUNoise
from myfm import FM, Net, NCF
from myfm import Predictor


def save_obj(obj, name):
	with open('../../data/new_ml_1M/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open('../../data/new_ml_1M/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Algorithm(object):
	def __init__(self, args, agent, predictor, env, data_list, target_list):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.args = args
		self.agent = agent
		self.ounoise = OUNoise(args.actor_output)
		self.predictor = predictor
		self.memory = ReplayMemory(args.memory_size)
		self.env = env

		self.train_data = torch.tensor(data_list.pop(0), dtype=torch.float32).to(self.device)
		self.train_target = torch.tensor(target_list.pop(0), dtype=torch.float32).to(self.device)
		self.valid_data = torch.tensor(data_list.pop(0), dtype=torch.float32).to(self.device)
		self.valid_target = torch.tensor(target_list.pop(0), dtype=torch.float32).to(self.device)

		train_data_set = Data.TensorDataset(self.train_data, self.train_target)
		shuffle = True if args.shuffle == 'y' else False
		print('shuffle train data...{}'.format(shuffle))
		self.train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=args.batch_size, shuffle=shuffle)


	def get_rmse(self, prediction, target):
		if prediction.shape != target.shape:
			prediction = prediction.squeeze()
		rmse = torch.sqrt(torch.sum((prediction - target)**2) / prediction.shape[0])
		return rmse.item()


	def evaluate(self, data, target, title='[Valid]'):
		self.agent.on_eval()
		self.predictor.on_eval()

		input_data = []
		for i_data, raw_feature in enumerate(data):
			state = self.env.get_history(raw_feature[0].item(), raw_feature[1].item())
			state = state.reshape((-1, state.shape[0], state.shape[1])).to(self.device)

			action = self.agent.select_action(state, action_noise=None)	# 不加噪声
			input_data.append(torch.cat([action.squeeze(), raw_feature]))

		input_data = torch.stack(input_data).to(self.device)
		prediction, predictor_loss = self.predictor.predict(input_data, target)
		rmse = self.get_rmse(prediction, target)
		reward = 0
		if self.args.reward == 'loss':
			# Average Reward e.g. Negative Average predictor loss
			reward = -(predictor_loss.mean().item() * self.args.alpha)
		elif self.args.reward == 'rmse':
			reward = -(rmse * self.args.alpha)

		print(title + ' RMSE:{:.6}, Average Reward:{:.8}'.format(
			rmse, reward))
		logging.info(title + ' RMSE:{:.6}, Average Reward:{:.8}'.format(
			rmse, reward))

		# self.agent.on_train()
		# self.predictor.on_train()
		return rmse


	def interactive(self, data, target):
		self.agent.on_eval()	# 切换为评估模式
		self.predictor.on_eval()

		for i_data, raw_feature in enumerate(data):
			mask = torch.tensor([True], dtype=torch.float32).to(self.device)
			state = self.env.get_history(raw_feature[0].item(), raw_feature[1].item())
			next_state = self.env.get_next_history(state, raw_feature[1].item(), 
				raw_feature[0].item(), target[i_data].item())

			# 转成符合 RNN 的输入数据形式
			state = state.reshape((-1, state.shape[0], state.shape[1])).to(self.device)
			next_state = next_state.reshape((-1, next_state.shape[0], next_state.shape[1])).to(self.device)

			action = self.agent.select_action(state, action_noise=self.ounoise)
			# action (1, 32) -> (32)
			input_data = torch.cat([action.squeeze(), raw_feature]).to(self.device)
			# input_data (32+22) -> (1, 32+22)
			input_data = input_data.reshape((1, input_data.shape[0])).to(self.device)
			# one_target (1, 1) 即:(batch=1, 1)
			one_target = torch.tensor([target[i_data]], dtype=torch.float32).reshape((1, 1)).to(self.device)
			# 先不训练
			prediction, predictor_loss = self.predictor.predict(input_data, one_target)
			# predictor loss 的负数作为 reward
			reward = 0
			if self.args.reward == 'loss':
				reward = torch.tensor([-(predictor_loss.item() * self.args.alpha)], dtype=torch.float32).to(self.device)
			elif self.args.reward == 'rmse':
				# rmse 的负数作为 reward
				reward = torch.tensor([-(self.get_rmse(prediction, one_target) * self.args.alpha)], dtype=torch.float32).to(self.device)

			self.memory.push(state, action, mask, next_state, reward)


	def train(self):
		rmse_list = []
		valid_rmse_list = []
		mean_predictor_loss_list = []
		for epoch in range(self.args.epoch):
			for i_batch, (data, target) in enumerate(self.train_data_loader):
				data = data.to(self.device)
				target = target.to(self.device)
				batch_state = []
				# 与 predictor 交互获得 agent 训练数据
				self.interactive(data.to(self.device), target.to(self.device))
				
				transitions = self.memory.sample(self.args.batch_size)
				batch = Transition(*zip(*transitions))
				self.agent.on_train()	# 切换为训练模式 (因为包含 BN\LN)
				value_loss, policy_loss = self.agent.update_parameters(batch)

				# 再训练 predictor
				self.agent.on_eval()	# 采集数据训练 predictor，所以切换为评估模式
				self.predictor.on_train()	# predictor 切换为训练模式
				input_data = []
				for i_data, raw_feature in enumerate(data):
					state = self.env.get_history(raw_feature[0].item(), raw_feature[1].item())
					state = state.reshape((-1, state.shape[0], state.shape[1])).to(self.device)

					action = self.agent.select_action(state, action_noise=None)	# 不加噪声
					input_data.append(torch.cat([action.squeeze(), raw_feature]).to(self.device))

				input_data = torch.stack(input_data).to(self.device)
				prediction, predictor_loss = self.predictor.train(input_data, target)
				rmse = self.get_rmse(prediction, target)

				predictor_loss_mean = predictor_loss.mean().item()
				mean_predictor_loss_list.append(predictor_loss_mean)
				rmse_list.append(rmse)
				reward = 0
				if self.args.reward == 'loss':
					# Average Reward e.g. Negative Average predictor loss
					reward = -(predictor_loss_mean * self.args.alpha)
				elif self.args.reward == 'rmse':
					reward = -(rmse * self.args.alpha)

				print('epoch:{}/{} i_batch:{}, RMSE:{:.6}, Average Reward:{:.8}'.format(epoch+1, self.args.epoch, 
					i_batch+1, rmse, reward), end = ', ')
				print('value loss:{:.4}, policy loss:{:.4}'.format(value_loss, policy_loss))
				logging.info('epoch:{}/{} i_batch:{}, RMSE:{:.6}, Average Reward:{:.8}'.format(epoch+1, self.args.epoch, 
					i_batch+1, rmse, reward))
			valid_rmse_list.append(self.evaluate(self.valid_data, self.valid_target))

		del self.train_data
		del self.train_target
		del self.valid_data
		del self.valid_target

		test_data = torch.tensor(np.load(self.args.base_data_dir + 'test_data.npy').astype(np.float32), dtype=torch.float32).to(self.device)
		test_target = torch.tensor(np.load(self.args.base_data_dir + 'test_target.npy').astype(np.float32), dtype=torch.float32).to(self.device)
		self.evaluate(test_data, test_target, title='[Test]')
		self.plot_result(rmse_list, valid_rmse_list, mean_predictor_loss_list)


	def pretrain_predictor(self):
		train_data_set = Data.TensorDataset(self.train_data, self.train_target)
		data_loader = Data.DataLoader(dataset=train_data_set, batch_size=self.args.batch_size, 
			shuffle=True)
		for epoch in range(self.args.pretrain_predictor_epoch):
			for i_batch, (data, target) in enumerate(data_loader):
				data = data.to(self.device)
				target = target.to(self.device)
				data_list = []
				# 补上 agent 输出的那部分为 0 向量
				for feature_vec in data:
					zero_vec = torch.zeros(self.args.actor_output).to(self.device)
					data_list.append(torch.cat([zero_vec, feature_vec]))

				data = torch.stack(data_list).to(self.device)
				prediction, loss = self.predictor.train(data, target)
				rmse = self.get_rmse(prediction, target)
				
				if (i_batch + 1) % 50 == 0:
					print('pretrain epoch:{}, i_batch:{}, loss:{:.5}, RMSE:{:.5}'.format(epoch + 1, 
						i_batch+1, loss.item(), rmse))


	def plot_result(self, rmse_list, valid_rmse_list, mean_predictor_loss_list):
		plt.figure(figsize=(8, 8))
		plt.subplot(1, 5, 1)
		plt.title('Train RMSE')
		plt.xlabel('Step')
		plt.ylabel('RMSE')
		plt.plot(rmse_list)

		plt.subplot(1, 5, 3)
		plt.title('Valid RMSE')
		plt.xlabel('Step')
		plt.ylabel('RMSE')
		plt.plot(valid_rmse_list)

		plt.subplot(1, 5, 5)
		plt.title('Mean Predictor LOSS')
		plt.xlabel('Step')
		plt.ylabel('LOSS')
		plt.plot(mean_predictor_loss_list)

		plt.savefig(self.args.base_pic_dir + self.args.v + '.png')
		if self.args.show_pic == 'y':
			plt.show()


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
		# stop_index = len(rating_list) - 1
		# for i, mid_rating_pair in enumerate(rating_list):
		# 	if curr_mid == mid_rating_pair[0]:
		# 		stop_index = i
		# 		break
		stop_index = self.index[uid][curr_mid]
		for i in range(stop_index - self.window, stop_index):
			if i < 0:
				history_feature = torch.zeros(23, dtype=torch.float32).to(self.device)
				history_feature[0] = uid
			else:
				mid = rating_list[i][0]
				rating  = rating_list[i][1]
				mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32).to(self.device)
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
		mfeature = torch.tensor(self.mid_map_mfeature[new_mid].astype(np.float32), dtype=torch.float32).to(self.device)
		rating = torch.tensor([rating], dtype=torch.float32).to(self.device)
		
		history_feature = torch.cat([torch.tensor([uid], dtype=torch.float32).to(self.device), mfeature, rating]).to(self.device)
		history_feature[-1] = (history_feature[-1] - self.rating_mean) / self.rating_std
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


def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for DDPG and Predictor")
	parser.add_argument('--v', default="v")
	parser.add_argument('--base_log_dir', default="../data/ddpg-fm/log/")
	parser.add_argument('--base_pic_dir', default="../data/ddpg-fm/pic/")
	parser.add_argument('--base_data_dir', default='../../data/new_ml_1M/')
	parser.add_argument('--memory_size', type=int, default=4096)
	parser.add_argument('--pretrain_predictor_epoch', type=int, default=100)
	parser.add_argument('--epoch', type=int, default=5)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--hw', type=int, default=10)	# history window
	parser.add_argument('--predictor', default='net')
	parser.add_argument('--pretrain', default='n')	# y -> pretrain predictor
	parser.add_argument('--reward', default='loss')
	parser.add_argument('--shuffle', default='y')
	parser.add_argument('--alpha', type=float, default=1)	# raw reward 乘以的倍数(试图放大 reward 加大训练幅度)
	# rating 范围
	parser.add_argument('--min', type=float, default=0.0)
	parser.add_argument('--max', type=float, default=5.0)

	parser.add_argument('--predictor_optim', default='adam')
	parser.add_argument('--actor_optim', default='adam')
	parser.add_argument('--critic_optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)	# sgd 时
	parser.add_argument('--norm_layer', default='bn')			# bn/ln/none
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
	parser.add_argument("--critic_lr", type=float, default=1e-3)
	parser.add_argument('--hidden_size', type=int, default=512)
	parser.add_argument('--actor_output', type=int, default=64)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--actor_tau', type=float, default=0.1)
	parser.add_argument('--critic_tau', type=float, default=0.1)
	parser.add_argument('--a_act', default='relu')
	parser.add_argument('--c_act', default='relu')
	# predictor
	parser.add_argument("--predictor_lr", type=float, default=1e-4)
	parser.add_argument('--n_act', default='relu')
	# embedding
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=64)
	parser.add_argument('--max_mid', type=int, default=193609)	# 1~193609
	parser.add_argument('--m_emb_dim', type=int, default=128)
	parser.add_argument('--g_emb_dim', type=int, default=16)	# genres emb dim
	# FM
	parser.add_argument('--fm_feature_size', type=int, default=22)	# 还要原来基础加上 actor_output
	parser.add_argument('--k', type=int, default=8)
	# network
	parser.add_argument('--hidden_0', type=int, default=1024)
	parser.add_argument('--hidden_1', type=int, default=512)
	# NCF
	parser.add_argument('--layers', default='1024,512,256')

	args = parser.parse_args()
	init_log(args)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('device:{}'.format(device))

	train_data = np.load(args.base_data_dir + 'train_data.npy').astype(np.float32)
	train_target = np.load(args.base_data_dir + 'train_target.npy').astype(np.float32)
	valid_data = np.load(args.base_data_dir + 'valid_data.npy').astype(np.float32)
	valid_target = np.load(args.base_data_dir + 'valid_target.npy').astype(np.float32)
	data_list = [train_data] + [valid_data]
	target_list = [train_target] + [valid_target]

	env = HistoryGenerator(args, device)
	agent = DDPG(args, device)

	# 后面还可以改成他的预测 rating 算法
	predictor_model = None
	if args.predictor == 'net':
		predictor_model = Net(args.u_emb_dim + args.m_emb_dim + args.g_emb_dim + args.actor_output, args.hidden_0, args.hidden_1, 1, args, device)
		print('predictor_model is Network.')
		logging.info('predictor_model is Network.')
	elif args.predictor == 'fm':
		predictor_model = FM(args.u_emb_dim + args.m_emb_dim + args.g_emb_dim + args.actor_output, args.k, args, device)
		print('predictor_model is FM.')
		logging.info('predictor_model is FM.')
	elif args.predictor ==  'ncf':
		predictor_model = NCF(args, args.u_emb_dim + args.actor_output + args.m_emb_dim + args.g_emb_dim, device)
		print('predictor_model is NCF.')
		logging.info('predictor_model is NCF.')

	predictor = Predictor(args, predictor_model, device)

	# 加载模型
	if args.load == 'y':
		agent.load_model(version=args.v)
		predictor.load(args.v)
		print('Loading version:{} models'.format(args.v))
		logging.info('Loading version:{} models'.format(args.v))

	algorithm = Algorithm(args, agent, predictor, env, data_list, target_list)
	if args.pretrain == 'y':
		# 暂时未发现有明显效果
		print('----------------------------------pretrain start----------------------------------')
		algorithm.pretrain_predictor()
		print('----------------------------------pretrain end----------------------------------')
		logging.info('pre-train.')
	else:
		print('no pre-train')
		logging.info('no pre-train.')
	algorithm.train()

	# 保存模型
	if args.save == 'y':
		algorithm.agent.save_model(version=args.v)
		predictor.save(args.v)
		print('Saving version:{} models'.format(args.v))
		logging.info('Saving version:{} models'.format(args.v))


if __name__ == '__main__':
	main()