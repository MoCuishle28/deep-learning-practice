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
from myfm import FM
from myfm import Predictor


def save_obj(obj, name):
	with open('../../data/new_ml_1M/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open('../../data/new_ml_1M/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Algorithm(object):
	def __init__(self, args, agent, predictor, env, data_list, target_list):
		self.args = args
		self.agent = agent
		self.ounoise = OUNoise(args.actor_output)
		self.predictor = predictor
		self.memory = ReplayMemory(args.memory_size)
		self.env = env

		self.train_data = torch.tensor(data_list.pop(0), dtype=torch.float32)
		self.train_target = torch.tensor(target_list.pop(0), dtype=torch.float32)
		self.valid_data = torch.tensor(data_list.pop(0), dtype=torch.float32)
		self.valid_target = torch.tensor(target_list.pop(0), dtype=torch.float32)
		self.test_data = torch.tensor(data_list.pop(0), dtype=torch.float32)
		self.test_target = torch.tensor(target_list.pop(0), dtype=torch.float32)
		self.process_data()

		train_data_set = Data.TensorDataset(self.train_data, self.train_target)
		self.train_data_loader = Data.DataLoader(dataset=train_data_set, batch_size=args.batch_size, shuffle=False)


	def process_data(self):
		# 将 uid, mid 加到前两维 然后标准化 (后续记得避开前两维再输入到 predictor)
		train_data, valid_data, test_data = [], [], []
		for x0 in self.train_data:
			train_data.append(torch.cat([torch.tensor([x0[0]]), torch.tensor([x0[1]]), x0]))
		for x1, x2 in zip(self.valid_data, self.test_data):
			valid_data.append(torch.cat([torch.tensor([x1[0]]), torch.tensor([x1[1]]), x1]))
			test_data.append(torch.cat([torch.tensor([x2[0]]), torch.tensor([x2[1]]), x2]))

		self.train_data = self.Standardization_uid_mid(torch.stack(train_data))
		self.valid_data = self.Standardization_uid_mid(torch.stack(valid_data))
		self.test_data = self.Standardization_uid_mid(torch.stack(test_data))


	def Standardization_uid_mid(self, data):
		uid_mean = data[:, 0].mean()
		uid_std = data[:, 0].std()
		mid_mean = data[:, 1].mean()
		mid_std = data[:, 1].std()
		data[:, 2] = (data[:, 2] - uid_mean) / uid_std
		data[:, 3] = (data[:, 3] - mid_mean) / mid_std
		return data


	def get_rmse(self, prediction, target):
		rmse = torch.sqrt(torch.sum((target - prediction)**2) / prediction.shape[0])
		return rmse.item()


	def evaluate(self, data, target, title='[Valid]'):
		input_data = []
		for i_data, raw_feature in enumerate(data):
			state = self.env.get_history(raw_feature[0].item(), raw_feature[1].item())
			state = state.reshape((-1, state.shape[0], state.shape[1]))

			action = self.agent.select_action(state, action_noise=None)	# 不加噪声
			# raw_feature 避开前两维没有标准化的 uid、mid
			input_data.append(torch.cat([action.squeeze(), raw_feature[2:]]))

		input_data = torch.stack(input_data)
		prediction, predictor_loss = self.predictor.predict(input_data, target)
		rmse = self.get_rmse(prediction, target)
		# Average Reward e.g. Negative Average predictor loss
		reward = -predictor_loss.mean().item()
		# reward = -rmse

		print(title + ' RMSE:{:.6}, Average Reward:{:.8}'.format(
			rmse, reward))
		logging.info(title + ' RMSE:{:.6}, Average Reward:{:.8}'.format(
			rmse, reward))
		return rmse



	def interactive(self, data, target):
		for i_data, raw_feature in enumerate(data):
			mask = torch.tensor([True], dtype=torch.float32)
			state = self.env.get_history(raw_feature[0].item(), raw_feature[1].item())
			next_state = self.env.get_next_history(state, raw_feature[1].item())

			# 转成符合 RNN 的输入数据形式
			state = state.reshape((-1, state.shape[0], state.shape[1]))
			next_state = next_state.reshape((-1, next_state.shape[0], next_state.shape[1]))

			action = self.agent.select_action(state, action_noise=self.ounoise)
			# action (1, 32) -> (32), raw_feature 避开前两维没有标准化的 uid、mid
			input_data = torch.cat([action.squeeze(), raw_feature[2:]])
			# input_data (32+22) -> (1, 32+22)
			input_data = input_data.reshape((1, input_data.shape[0]))
			# one_target (1, 1) 即:(batch=1, 1)
			one_target = torch.tensor([target[i_data]], dtype=torch.float32).reshape((1, 1))
			# 先不训练 (或者考虑如何预训练？ TODO)
			prediction, predictor_loss = self.predictor.predict(input_data, one_target)
			# predictor loss 的负数作为 reward
			reward = torch.tensor([-predictor_loss.item()], dtype=torch.float32)
			# rmse 的负数作为 reward
			# reward = torch.tensor([-self.get_rmse(prediction, one_target)], dtype=torch.float32)

			self.memory.push(state, action, mask, next_state, reward)



	def train(self):
		rmse_list = []
		valid_rmse_list = []
		mean_predictor_loss_list = []
		for epoch in range(self.args.epoch):
			for i_batch, (data, target) in enumerate(self.train_data_loader):
				batch_state = []
				# 与 predictor 交互获得 agent 训练数据
				self.interactive(data, target)
				
				transitions = self.memory.sample(self.args.batch_size)
				batch = Transition(*zip(*transitions))
				value_loss, policy_loss = self.agent.update_parameters(batch)

				# 再训练 predictor
				input_data = []
				for i_data, raw_feature in enumerate(data):
					state = self.env.get_history(raw_feature[0].item(), raw_feature[1].item())
					state = state.reshape((-1, state.shape[0], state.shape[1]))

					action = self.agent.select_action(state, action_noise=None)	# 不加噪声
					# raw_feature 避开前两维没有标准化的 uid、mid
					input_data.append(torch.cat([action.squeeze(), raw_feature[2:]]))

				input_data = torch.stack(input_data)
				prediction, predictor_loss = self.predictor.train(input_data, target)
				rmse = self.get_rmse(prediction, target)

				predictor_loss_mean = -predictor_loss.mean().item()
				mean_predictor_loss_list.append(predictor_loss_mean)
				rmse_list.append(rmse)
				# Average Reward e.g. Negative Average predictor loss
				reward = predictor_loss_mean
				# reward = -rmse

				print('epoch:{}/{} i_batch:{}, RMSE:{:.6}, Average Reward:{:.8}'.format(epoch+1, self.args.epoch, 
					i_batch+1, rmse, reward), end = ', ')
				print('value loss:{:.4}, policy loss:{:.4}'.format(value_loss, policy_loss))
				logging.info('epoch:{}/{} i_batch:{}, RMSE:{:.6}, Average Reward:{:.8}'.format(epoch+1, self.args.epoch, 
					i_batch+1, rmse, reward))

			valid_rmse_list.append(self.evaluate(self.valid_data, self.valid_target))
		self.evaluate(self.test_data, self.test_target, title='[Test]')
		self.plot_result(rmse_list, valid_rmse_list, mean_predictor_loss_list)


	def plot_result(self, rmse_list, valid_rmse_list, mean_predictor_loss_list):
		plt.figure(figsize=(15, 8))
		plt.subplot(5, 1, 1)
		plt.title('Train RMSE')
		plt.xlabel('Step')
		plt.ylabel('RMSE')
		plt.plot(rmse_list)

		plt.subplot(5, 1, 3)
		plt.title('Valid RMSE')
		plt.xlabel('Step')
		plt.ylabel('RMSE')
		plt.plot(valid_rmse_list)

		plt.subplot(5, 1, 5)
		plt.title('Mean Predictor LOSS')
		plt.xlabel('Step')
		plt.ylabel('LOSS')
		plt.plot(mean_predictor_loss_list)

		plt.show()


class HistoryGenerator(object):
	def __init__(self, args):
		self.args = args
		# mid: one-hot feature (21维 -> mid, genre, genre, ...)
		self.mid_map_mfeature = load_obj('mid_map_mfeature')		
		self.users_rating = load_obj('users_rating_without_timestamp') # uid:[[mid, rating], ...] 有序
		# self.users_has_clicked = load_obj('users_has_clicked')	# uid:{mid, mid, ...}
		self.window = args.history_window


	def get_history(self, uid, curr_mid):
		'''
		return: tensor (window, feature size:23 dim)
		'''
		ret_data = []
		rating_list = self.users_rating[uid]
		stop_index = len(rating_list) - 1
		for i, mid_rating_pair in enumerate(rating_list):
			if curr_mid == mid_rating_pair[0]:
				stop_index = i
				break
		for i in range(stop_index - self.window, stop_index):
			if i < 0:
				history_feature = torch.zeros(23, dtype=torch.float32)
				history_feature[0] = uid
			else:
				mid = rating_list[i][0]
				rating  = rating_list[i][1]
				mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32)
				# [uid, mfeature..., rating]
				history_feature = torch.cat([torch.tensor([uid], dtype=torch.float32), 
					mfeature, 
					torch.tensor([rating], dtype=torch.float32)])
			ret_data.append(history_feature)
		return torch.stack(ret_data)


	def get_next_history(self, curr_history, new_mid):
		'''
		这个 state 的转移方式没有考虑 action(即: trasition probability = 1) TODO
		curr_history: tensor (window, feature size)
		return: tensor (window, feature size)
		'''
		curr_history = curr_history.tolist()
		curr_history.pop(0)
		uid = curr_history[0][0]
		rating = -1
		for behavior in self.users_rating[uid]:
			if new_mid == behavior[0]:
				rating = behavior[1]
				break
		mfeature = torch.tensor(self.mid_map_mfeature[new_mid].astype(np.float32), dtype=torch.float32)
		history_feature = torch.cat([torch.tensor([uid], dtype=torch.float32), 
					mfeature, 
					torch.tensor([rating], dtype=torch.float32)])
		curr_history.append(history_feature)
		return torch.tensor(curr_history)
		

def init_log(args):
	start = datetime.datetime.now()
	logging.basicConfig(level = logging.INFO,
					filename = args.base_log_dir + str(time.time()) + '.log',
					filemode = 'a',		# 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
					# a是追加模式，默认如果不写的话，就是追加模式
					)
	logging.info('start! '+str(start))
	logging.info('Parameter:')
	logging.info(str(args))
	logging.info('\n-------------------------------------------------------------\n')


def main():
	seq_output_size = 32
	actor_output = 32

	parser = argparse.ArgumentParser(description="Hyperparameters for DDPG and FM")
	parser.add_argument('--base_log_dir', default="../data/ddpg-fm/log/")
	parser.add_argument('--base_data_dir', default='../../data/new_ml_1M/')
	parser.add_argument('--memory_size', type=int, default=4096)
	parser.add_argument('--epoch', type=int, default=5)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--history_window', type=int, default=5)
	# seq model
	parser.add_argument('--seq_input_size', type=int, default=23)
	parser.add_argument('--seq_hidden_size', type=int, default=64)
	parser.add_argument('--seq_layer_num', type=int, default=2)
	parser.add_argument('--seq_output_size', type=int, default=seq_output_size)
	# ddpg
	parser.add_argument("--actor_lr", type=float, default=1e-4)
	parser.add_argument("--critic_lr", type=float, default=1e-4)
	parser.add_argument('--num_input', type=int, default=seq_output_size)	# 等于 seq_output_size
	parser.add_argument('--hidden_size', type=int, default=128)
	parser.add_argument('--actor_output', type=int, default=actor_output)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--tau', type=float, default=0.01)
	# FM
	parser.add_argument("--fm_lr", type=float, default=1e-3)
	parser.add_argument('--fm_feature_size', type=int, default=22+actor_output)	# 原来基础加上 actor_output
	parser.add_argument('--k', type=int, default=20)
	args = parser.parse_args()
	init_log(args)

	train_data = np.load(args.base_data_dir + 'train_data.npy').astype(np.float32)
	train_target = np.load(args.base_data_dir + 'train_target.npy').astype(np.float32)
	valid_data = np.load(args.base_data_dir + 'valid_data.npy').astype(np.float32)
	valid_target = np.load(args.base_data_dir + 'valid_target.npy').astype(np.float32)
	test_data = np.load(args.base_data_dir + 'test_data.npy').astype(np.float32)
	test_target = np.load(args.base_data_dir + 'test_target.npy').astype(np.float32)
	data_list = [train_data] + [valid_data] + [test_data]
	target_list = [train_target] + [valid_target] + [test_target]

	env = HistoryGenerator(args)

	agent = DDPG(args)
	# 后面还可以改成 nn 或其他的预测 rating 算法
	predictor = Predictor(args, FM(args.fm_feature_size, args.k))
	algorithm = Algorithm(args, agent, predictor, env, data_list, target_list)
	algorithm.train()


if __name__ == '__main__':
	main()