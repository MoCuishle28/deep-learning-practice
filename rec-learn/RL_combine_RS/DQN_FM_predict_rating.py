import pickle
import argparse
import random
import logging
import datetime
import time

import torch
import torch.nn as nn
import numpy as np
import gym
from collections import deque
import matplotlib.pyplot as plt


def save_obj(obj, name):
	with open('../data/ml20/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open('../data/ml20/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class FM(nn.Module):
	def __init__(self, feature_size, k):
		super(FM, self).__init__()
		self.w0 = nn.Parameter(torch.empty(1, dtype=torch.float32))
		nn.init.normal_(self.w0)

		# 不加初始化会全 0
		self.w1 = nn.Parameter(torch.empty(feature_size, 1, dtype=torch.float32))
		nn.init.xavier_normal_(self.w1)

		# 不加初始化会全 0
		self.v = nn.Parameter(torch.empty(feature_size, k, dtype=torch.float32))
		nn.init.xavier_normal_(self.v)


	def forward(self, X):
		'''
		X: (batch, feature_size)
		'''
		inter_1 = torch.mm(X, self.v)
		inter_2 = torch.mm((X**2), (self.v**2))
		interaction = (0.5*torch.sum((inter_1**2) - inter_2, dim=1)).reshape(X.shape[0], 1)
		predict = self.w0 + torch.mm(X, self.w1) + interaction
		return predict


class StateModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size):
		super(StateModel, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

		
	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		# c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		
		out, _ = self.lstm(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
		out = self.fc(out[:, -1, :])	# 最后时刻的 seq 作为输出
		return out


class Network(nn.Module):
	def __init__(self, state_size, hidden_size0, hidden_size1, output_size):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(state_size, hidden_size0)
		self.fc2 = nn.Linear(hidden_size0, hidden_size1)
		self.fc3 = nn.Linear(hidden_size1, output_size)


	def forward(self, state):
		hidden = torch.relu(self.fc1(state))
		hidden = torch.relu(self.fc2(hidden))
		return self.fc3(hidden)


class DQN_combine_FM(object):
	def __init__(self, args, env, train_data, train_target, valid_data, valid_target, test_data, test_target):
		self.args = args
		self.env = env
		self.state_model = StateModel(args.state_size, args.state_model_hidden_size, args.state_model_layer_num, args.state_size)
		self.network = Network(args.state_size, args.hidden_size0, args.hidden_size1, args.output_size)

		state_model_params = [param for param in self.state_model.parameters()]
		q_network_params = [param for param in self.network.parameters()]
		parameters_list = state_model_params + q_network_params
		self.optimizer = torch.optim.Adam(parameters_list, lr=args.dqn_lr)

		self.lossFunc = nn.MSELoss()
		self.replay_buffer = deque(maxlen=args.maxlen)

		self.fm = FM(args.fm_feature_size, args.k)
		self.fm_optimize = torch.optim.Adam(self.fm.parameters(), lr=args.fm_lr)
		self.fm_criterion = nn.MSELoss()

		self.train_data = train_data
		self.train_target = train_target
		self.valid_data = valid_data
		self.valid_target = valid_target
		self.test_data = test_data
		self.test_target = test_target


	def save_model(self):
		# TODO
		pass


	def evaluate(self, data, target):
		batch_state = []
		batch_action = []
		fm_input_data = []

		for x in data:
			state = self.env.get_history_embedding(x[0], x[1])

			action = self.network(
				self.state_model(torch.tensor(state, dtype=torch.float32))).detach().numpy()
			action = np.argmax(action)
		
			# 若分类正确则用户点击了, next_state: 加上点击后的 embedding
			genres_one_hot = np.zeros(self.args.output_size, dtype=np.float32)
			genres_one_hot[action] = 1
			x = np.concatenate([x, genres_one_hot])		# fm 的输入
			x = self.normalize_uid_mid(x)
			fm_input_data.append(x)		# list

			batch_state.append(state)
			batch_action.append(action)

		predict, _ = self.train_fm(fm_input_data, target)
		predict = torch.tensor(predict, dtype=torch.float32)
		target = torch.tensor(target, dtype=torch.float32).reshape((target.shape[0], 1))
		rmse = torch.sqrt(torch.sum((target - predict)**2) / predict.shape[0])
		return rmse.item()


	def train_fm(self, fm_input_data, target):
		fm_input_data = torch.tensor(fm_input_data, dtype=torch.float32)
		predict = self.fm(fm_input_data)
		target = torch.tensor(target, dtype=torch.float32).reshape((target.shape[0], 1))
		loss = self.fm_criterion(predict, target)

		self.fm_optimize.zero_grad()
		loss.backward()
		self.fm_optimize.step()

		return predict.detach().numpy(), loss.item()


	def get_batch(self, batch):
		for i in range(0, self.train_data.shape[0], batch):
			if i+batch < self.train_data.shape[0]:
				yield self.train_data[i:i+batch, :], self.train_target[i:i+batch]
			else:
				yield self.train_data[i:, :], self.train_target[i:]


	def train_DQN(self):
		train_rmse_list = []
		DQN_loss_list = []
		fm_loss_list = []
		valid_rmse_list = []
		for epoch_i in range(self.args.epoch):
			# 计算当前探索率
			epsilon = max(
				self.args.initial_epsilon * 
				(self.args.num_exploration_episodes - epoch_i) / self.args.num_exploration_episodes, 
				self.args.final_epsilon)

			for data, target in self.get_batch(self.args.batch_size):
				batch_state = []
				batch_next_state = []
				batch_action = []
				fm_input_data = []

				for x in data:
					state = self.env.get_history_embedding(x[0], x[1])

					if random.random() < epsilon:
						action = random.choice(range(self.args.output_size))	# 0~18 选一个
					else:
						action = self.network(
							self.state_model(torch.tensor(state, dtype=torch.float32))).detach().numpy()
						action = np.argmax(action)
				
					# 若分类正确则用户点击了, next_state: 加上点击后的 embedding
					genres_one_hot = np.zeros(self.args.output_size, dtype=np.float32)
					genres_one_hot[action] = 1
					x = np.concatenate([x, genres_one_hot])		# fm 的输入
					x = self.normalize_uid_mid(x)
					fm_input_data.append(x)		# list

					batch_state.append(state)
					batch_action.append(action)

				predict, fm_loss = self.train_fm(fm_input_data, target)
				fm_loss_list.append(fm_loss)

				target = target.reshape((target.shape[0], 1))
				error = -np.abs(target - predict)	# 负的误差作为 reward
				batch_reward = error[:, 0]		# 要只有一维的

				# 如果没有点击, 则 state 不变
				for curr_state, x in zip(batch_state, data):
					batch_next_state.append(self.env.get_next_state(curr_state, x[1]))
				
				# (state, action, reward, next_state) 4 元组
				for state, action, reward, next_state in zip(batch_state, batch_action, batch_reward, batch_next_state):
					self.replay_buffer.append((state, action, reward, next_state))

				train_rmse = self.evaluate(data, target)
				train_rmse_list.append(train_rmse)

				# 太慢 TODO
				valid_rmse = 0
				# valid_rmse = self.evaluate(self.valid_data, self.valid_target)
				# valid_rmse_list.append(valid_rmse)

				if len(self.replay_buffer) >= self.args.batch_size:
					# 从经验回放池中随机取一个批次的 4 元组 
					batch_state, batch_action, batch_reward, batch_next_state = zip(
						*random.sample(self.replay_buffer, self.args.batch_size))

					# 转换为 torch.Tensor
					batch_state, batch_reward, batch_next_state = \
						[torch.tensor(a, dtype=torch.float32) 
						for a in [batch_state, batch_reward, batch_next_state]]
					batch_action = torch.LongTensor(batch_action).view(len(batch_action), 1)

					batch_next_state = batch_next_state.reshape((batch_next_state.shape[0], batch_next_state.shape[2], batch_next_state.shape[3]))
					batch_state = batch_state.reshape((batch_state.shape[0], batch_state.shape[2], batch_state.shape[3]))

					q_value = self.network(self.state_model(batch_next_state))
					next_predict = batch_reward + (self.args.gamma * torch.max(q_value, dim=-1).values)

					curr_state_q_value = self.network(self.state_model(batch_state))
					# (batch, action space)
					one_hot_act = torch.zeros(self.args.batch_size, self.args.output_size).scatter_(dim=1, index=batch_action, value=1)
					curr_predict = torch.sum(curr_state_q_value * one_hot_act, dim=-1)

					# 最小化对下一步 Q-value 的预测和当前对 Q-value 的预测的差距 (TD)
					loss = self.lossFunc(next_predict, curr_predict)
					DQN_loss_list.append(loss.item())

					print(('Epoch:{}/{}, Train RMSE:{:.4f}, Valid RMSE:{:.4f}, ' + 
							'FM Loss:{:.4f}, DQN Loss:{:.8f}').format(
							self.args.epoch, epoch_i+1, 
							train_rmse, valid_rmse, fm_loss, loss.item()))
					logging.debug(('Train RMSE:{:.4f}, Valid RMSE:{:.4f}, ' + 
							'FM Loss:{:.4f}, DQN Loss:{:.8f}').format(
							train_rmse, valid_rmse, fm_loss, loss.item()))
					
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()

			# 每训练一个 epoch 评估一次
			valid_rmse = self.evaluate(self.valid_data, self.valid_target)
			valid_rmse_list.append(valid_rmse)
			logging.debug('Valid RMSE:{:.4f}'.format(valid_rmse))
			print('Valid RMSE:{:.4f}'.format(valid_rmse))

		test_precise = self.evaluate(self.test_data, self.test_target)
		print('TEST RMSE:{:.4f}'.format(test_precise))
		logging.debug('TEST RMSE:{:.4f}'.format(test_precise))
		self.plot_loss_reward(DQN_loss_list, fm_loss_list, train_rmse_list, valid_rmse_list)


	def normalize_uid_mid(self, data):
		data[0] = (data[0] - 1) / (610 - 1)
		data[1] = (data[1] - 1) / (117590 - 1)
		return data


	def plot_loss_reward(self, DQN_loss_list, fm_loss_list, train_rmse_list, valid_rmse_list):
		plt.figure(figsize=(15, 8))
		plt.subplot(5, 1, 1)
		plt.title('DQN LOSS')
		plt.xlabel('Step')
		plt.ylabel('LOSS')
		plt.plot([i for i in range(len(DQN_loss_list))], DQN_loss_list)

		plt.subplot(5, 1, 3)
		plt.title('FM LOSS')
		plt.xlabel('Step')
		plt.ylabel('LOSS')
		plt.plot([i for i in range(len(fm_loss_list))], fm_loss_list)

		plt.subplot(5, 1, 5)
		plt.title('RMSE')
		plt.xlabel('Step')
		plt.ylabel('RMSE')
		train_line, = plt.plot([i for i in range(len(train_rmse_list))], train_rmse_list, label='train RMSE', color='blue')
		valid_line, = plt.plot([i for i in range(len(valid_rmse_list))], valid_rmse_list, label='valid RMSE', color='red')
		plt.legend(handles=[train_line, valid_line], loc='best')

		plt.show()


class GenerateHistoryEmbedding(object):
	def __init__(self, args):
		self.movie_embedding_128_mini = load_obj('mini_ml20_mid_map_one_hot')	# mid:one-hot (20 维)
		self.users_behavior = load_obj('users_rating')	# uid:[[mid, rating, timestamp], ...] 有序
		self.users_has_clicked = load_obj('users_has_clicked_mini')

		self.args = args
		self.window = 5					# 考虑最近多少部电影, 不够补 0 向量


	def get_history_embedding(self, uid, curr_mid):
		# 面对负样本时, 随机选一个正样本的 mid 作为时间点 (暂时这么处理 TODO)
		if curr_mid not in self.users_has_clicked[uid]:
			curr_mid = random.sample(self.users_has_clicked[uid], 1)[0]

		history_embedding = []		# (window, 128)
		for item in self.users_behavior[uid]:
			if item[0] != curr_mid:
				if len(history_embedding) == self.window:
					history_embedding.pop()
				# 加上 uid
				new_one_hot = np.concatenate([np.array([uid]), self.movie_embedding_128_mini[item[0]]])
				history_embedding.append(new_one_hot)
			else:
				break

		while len(history_embedding) < self.window:
			history_embedding.insert(0, np.concatenate([np.array([uid]), np.zeros(20, dtype=np.float32)]))
		# (batch:1, seq:window, embedding_size:128)
		input_data = np.stack(history_embedding)
		input_data = input_data.reshape((1, len(history_embedding), self.args.state_size))
		return input_data


	def get_next_state(self, curr_state, add_mid):
		curr_state = curr_state.tolist()
		curr_state[0].pop(0)
		# 加上 uid
		new_one_hot = np.concatenate([np.array([curr_state[0][0][0]]), self.movie_embedding_128_mini[add_mid]])
		curr_state[0].append(new_one_hot)
		return np.array(curr_state)


def generate_valid_test_data(data, target):
	'''
	8:1:1
	return: train_data, train_target, valid_data, valid_target, test_data, test_target
	'''
	unit = data.shape[0] // 10
	train_size, valid_size, test_size = data.shape[0] - 2*unit, unit, unit
	all_index_list = [i for i in range(data.shape[0])]

	valid_index = random.sample(all_index_list, valid_size)
	sample_set = set(valid_index)
	all_index_list = [i for i in all_index_list if i not in sample_set]		# 删去已经抽出的数据索引
	
	test_index = random.sample(all_index_list, test_size)
	sample_set = set(test_index)
	all_index_list = [i for i in all_index_list if i not in sample_set]

	random.shuffle(all_index_list)
	train_index = all_index_list

	data = data.tolist()
	target = target.tolist()

	train_data, train_target = np.array([data[i] for i in train_index]), np.array([target[i] for i in train_index])
	valid_data, valid_target = np.array([data[i] for i in valid_index]), np.array([target[i] for i in valid_index])
	test_data, test_target = np.array([data[i] for i in test_index]), np.array([target[i] for i in test_index])

	return train_data, train_target, valid_data, valid_target, test_data, test_target


def init_log(args):
	start = datetime.datetime.now()
	logging.basicConfig(level = logging.DEBUG,			# 控制台打印的日志级别
					filename = "data/log/predict-rating-"+str(time.time())+'.log',
					filemode = 'a',					# 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
					# a是追加模式，默认如果不写的话，就是追加模式
					)
	logging.debug('start! '+str(start))
	logging.debug('Parameter: ')
	logging.debug('fm_lr:' + str(args.fm_lr) + ', dqn_lr:' + str(args.dqn_lr) + ', epoch:'+str(args.epoch))
	logging.debug('batch_size:'+ str(args.batch_size) + ', state_size:' + str(args.state_size) + ', state_model_hidden_size:' + str(args.state_model_hidden_size))
	logging.debug('state_model_layer_num:' + str(args.state_model_layer_num) + ', hidden_size0:' + str(args.hidden_size0))
	logging.debug('hidden_size1:'+ str(args.hidden_size1) + ', output_size:' + str(args.output_size) + ', gamma:' + str(args.gamma))
	logging.debug('num_exploration_episodes:' + str(args.num_exploration_episodes) + ', initial_epsilon:' + str(args.initial_epsilon) + ', final_epsilon:' + str(args.final_epsilon))
	logging.debug('maxlen:' + str(args.maxlen) + ', fm_feature_size:' + str(args.fm_feature_size) + ', k:' + str(args.k))
	logging.debug('-------------------------------------------------------------')


def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for Q-Learning and FM")
	parser.add_argument("--fm_lr", type=float, default=1e-2)
	parser.add_argument("--dqn_lr", type=float, default=1e-3)
	parser.add_argument('--epoch', type=int, default=5)
	parser.add_argument('--batch_size', type=int, default=1024)

	parser.add_argument('--state_size', type=int, default=21)
	parser.add_argument('--state_model_hidden_size', type=int, default=128)
	parser.add_argument('--state_model_layer_num', type=int, default=2)
	parser.add_argument('--hidden_size0', type=int, default=256)
	parser.add_argument('--hidden_size1', type=int, default=512)
	parser.add_argument('--output_size', type=int, default=19)		# gernes number 0~18
	parser.add_argument('--gamma', type=float, default=0.95)
	# 探索过程所占的episode数量
	parser.add_argument('--num_exploration_episodes', type=int, default=500)
	parser.add_argument('--initial_epsilon', type=float, default=1.0)	# 探索起始时的探索率
	parser.add_argument('--final_epsilon', type=float, default=0.01)	# 探索终止时的探索率
	parser.add_argument('--maxlen', type=int, default=4096)

	parser.add_argument('--fm_feature_size', type=int, default=40)	# 在原来基础上加上 DQN 选的 gerne one-hot
	parser.add_argument('--k', type=int, default=20)
	args = parser.parse_args()

	init_log(args)


	# 正样本是按照时间顺序的
	data = np.load('../data/ml20/mini_data.npy').astype(np.float32)
	target = np.load('../data/ml20/mini_target.npy').astype(np.int8)

	# 划分数据集
	train_data, train_target, valid_data, valid_target, test_data, test_target = generate_valid_test_data(data, target)
	del data
	del target

	generator = GenerateHistoryEmbedding(args)
	model = DQN_combine_FM(args, generator, train_data, train_target, valid_data, valid_target, test_data, test_target)
	model.train_DQN()

	logging.debug(datetime.datetime.now())
	logging.debug('---------------------END---------------------')


if __name__ == '__main__':
	main()