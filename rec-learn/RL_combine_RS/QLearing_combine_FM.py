import pickle
import argparse
import random

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
		return torch.sigmoid(predict)


class StateModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size):
		super(StateModel, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

		
	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
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
	def __init__(self, args, env, data, target):
		self.args = args
		self.env = env
		self.state_model = StateModel(args.state_size, args.state_model_hidden_size, args.state_model_layer_num, args.state_size)
		self.network = Network(args.state_size, args.hidden_size0, args.hidden_size1, args.output_size)

		state_model_params = [param for param in self.state_model.parameters()]
		q_network_params = [param for param in self.network.parameters()]
		parameters_list = state_model_params + q_network_params
		self.optimizer = torch.optim.Adam(parameters_list, lr=args.lr)

		self.lossFunc = nn.MSELoss()
		self.replay_buffer = deque(maxlen=args.maxlen)

		self.fm = FM(args.feature_size, args.k)
		self.fm_optimize = torch.optm.Adam(fm.parameters(), lr=args.lr)
		self.fm_criterion = nn.BCELoss()	# 是否点击 (CTR 预测)
		self.data = data
		self.target = target


	def save_model(self):
		# TODO
		pass


	def train_fm(self):
		# TODO
		pass


	def train_DQN(self):
		# TODO 应该从 data 中取数据训练
		reward_list = []
		loss_list = []
		for episode_i in range(self.args.epoch):
			# 计算当前探索率
			epsilon = max(
				self.args.initial_epsilon * 
				(self.args.num_exploration_episodes - episode_i) / self.args.num_exploration_episodes, 
				self.args.final_epsilon)

			total_reward = 0
			while True:
				if random.random() < epsilon:
					action = random.choice(range(self.args.output_size))	# 0~18 选一个
				else:
					action = self.network(self.state_model(state)).detach().numpy()
					action = np.argmax(action)
				# 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
				next_state, reward, done = self.env.step(action)
				
				# (state, action, reward, next_state, done) 5 元组
				self.replay_buffer.append((state.detach().numpy(), action, reward, next_state.detach().numpy(), 1 if done else 0))
				state = next_state
				total_reward += reward

				if done:
					print('Episode:{}/{} Total Reward:{}'.format(self.args.epoch, episode_i+1, 
						total_reward))
					reward_list.append(total_reward)
					break

				if len(self.replay_buffer) >= self.args.batch_size:
					# 从经验回放池中随机取一个批次的 5 元组 
					batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
						*random.sample(self.replay_buffer, self.args.batch_size))

					# 转换为 torch.Tensor
					batch_state, batch_reward, batch_next_state, batch_done = \
						[torch.tensor(a, dtype=torch.float32) 
						for a in [batch_state, batch_reward, batch_next_state, batch_done]]
					batch_action = torch.LongTensor(batch_action).view(len(batch_action), 1)

					batch_next_state = batch_next_state.reshape((batch_next_state.shape[0], batch_next_state.shape[2], batch_next_state.shape[3]))
					batch_state = batch_state.reshape((batch_state.shape[0], batch_state.shape[2], batch_state.shape[3]))

					q_value = self.network(self.state_model(batch_next_state))
					next_predict = batch_reward + (self.args.gamma * torch.max(q_value, dim=-1).values) * (1 - batch_done)

					curr_state_q_value = self.network(self.state_model(batch_state))
					# (batch, action space)
					one_hot_act = torch.zeros(self.args.batch_size, self.args.output_size).scatter_(dim=1, index=batch_action, value=1)
					curr_predict = torch.sum(curr_state_q_value * one_hot_act, dim=-1)

					# 最小化对下一步 Q-value 的预测和当前对 Q-value 的预测的差距 (TD)
					loss = self.lossFunc(next_predict, curr_predict)
					loss_list.append(loss)
					
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()


class GenerateHistoryEmbedding(object):
	def __init__(self):
		self.movie_embedding_128_mini = load_obj('movie_embedding_128_mini')	# mid:embedding
		self.users_behavior = load_obj('users_rating')	# uid:[[mid, rating, timestamp], ...] 有序
		self.users_has_clicked = load_obj('users_has_clicked_mini')

		# uid:{mid:rating, ...}
		# self.users_rating = {k:{x[0]:x[1] for x in v} for k,v in self.users_behavior.items()}
		# self.uid_map_uRow = load_obj('uid_map_uRow')
		# self.uRow_map_uid = {uRow:uid for uid, uRow in self.uid_map_uRow.items()}
		# self.mid_map_mRow = load_obj('mid_map_mRow')
		# self.mRow_map_mid = {mRow:mid for mid, mRow in self.mid_map_mRow.items()}

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
				history_embedding.append(self.movie_embedding_128_mini[item[0]])
			else:
				break

		while len(history_embedding) < self.window:
			history_embedding.insert(0, torch.zeros(128, dtype=torch.float32))
		# (batch:1, seq:window, embedding_size:128)
		input_data = torch.stack(history_embedding).reshape((1, len(history_embedding), 128))
		return input_data



def generate_negative_samples(data, target):
	users_rating = load_obj('users_rating')
	movie_genres = load_obj('movie_genres')		# mid:[genres idx, genres idx, ...]
	# uid:{mid, mid, ...}
	users_has_clicked = {
		uid : set([item[0] for item in item_list]) for uid, item_list in users_rating.items()}
	del users_rating

	target[:] = 1		# 用于分类问题, 把 rating 改为 0/1
	data = data.tolist()
	target = target.tolist()
	all_mid = set(movie_genres.keys())

	# 产生负样本的数量要等于正样本
	for uid, mid_set in users_has_clicked.items():
		sample_set = all_mid - mid_set
		negative_mid_list = random.sample(sample_set, len(mid_set))
		for mid in negative_mid_list:
			x = np.zeros(21, dtype=np.int)
			x[0], x[1] = uid, mid			# uid, mid
			genes_list = movie_genres[mid]
			for idx in genes_list:
				x[idx+2] = 1

			data.append(x)
			target.append(0)

	data = np.array(data)
	target = np.array(target)
	return data, target


def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for Q-Learning and FM")
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument('--epoch', type=int, default=1000)

	parser.add_argument('--state_size', type=int, default=128)
	parser.add_argument('--state_model_hidden_size', type=int, default=256)
	parser.add_argument('--state_model_layer_num', type=int, default=2)
	parser.add_argument('--hidden_size0', type=int, default=256)
	parser.add_argument('--hidden_size1', type=int, default=512)
	parser.add_argument('--output_size', type=int, default=19)		# gernes number 0~18
	parser.add_argument('--gamma', type=float, default=0.95)
	# 探索过程所占的episode数量
	parser.add_argument('--num_exploration_episodes', type=int, default=500)
	parser.add_argument('--initial_epsilon', type=float, default=1.0)	# 探索起始时的探索率
	parser.add_argument('--final_epsilon', type=float, default=0.01)	# 探索终止时的探索率
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--maxlen', type=int, default=1000)

	parser.add_argument('--fm_feature_size', type=int, default=23)	# 在原来基础上加上 DQN 选的 gerne rating
	parser.add_argument('--k', type=int, default=10)
	args = parser.parse_args()

	# # 构造带负样本的数据集
	# data = np.load('../data/ml20/mini_data.npy').astype(np.float32)
	# target = np.load('../data/ml20/mini_target.npy')
	# data, target = generate_negative_samples(data, target)
	# np.save('../data/ml20/mini_data_with_negative.npy', data)
	# np.save('../data/ml20/mini_target_with_negative.npy', target)

	# 正样本是按照时间顺序的
	data = np.load('../data/ml20/mini_data_with_negative.npy').astype(np.float32)
	target = np.load('../data/ml20/mini_target_with_negative.npy').astype(np.int8)
	
	generator = GenerateHistoryEmbedding()
	model = DQN_combine_FM(args, generator, data, target)
	# 训练 TODO
	


if __name__ == '__main__':
	main()