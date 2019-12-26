import pickle
import argparse
import random

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from movieLens_env import MovieLens_Env


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
		self.fc = nn.Linear(hidden_size0, hidden_size1)
		self.fc2 = nn.Linear(hidden_size1, output_size)


	def forward(self, state):
		hidden = torch.relu(self.fc1(state))
		hidden = torch.relu(self.fc(hidden))
		return self.fc2(hidden)


class Q_Learning(object):
	def __init__(self, args, env):
		self.args = args
		self.env = env
		self.state_model = StateModel(128, 256, 2, 128)
		self.network = Network(args.state_size, args.hidden_size0, args.hidden_size1, args.output_size)

		state_model_params = [param for param in self.state_model.parameters()]
		q_network_params = [param for param in self.network.parameters()]
		parameters_list = state_model_params + q_network_params
		self.optimizer = torch.optim.Adam(parameters_list, lr=args.lr)

		self.lossFunc = nn.MSELoss()
		self.replay_buffer = deque(maxlen=args.maxlen)


	def save_model(self):
		pass


	def train(self):
		reward_list = []
		loss_list = []
		for episode_i in range(self.args.epoch):
			state = self.env.reset()
			# 计算当前探索率
			epsilon = max(
				self.args.initial_epsilon * 
				(self.args.num_exploration_episodes - episode_i) / self.args.num_exploration_episodes, 
				self.args.final_epsilon)

			total_reward = 0
			while True:
				# if self.args.render:
				# 	self.env.render()

				if random.random() < epsilon:
					action = self.env.sample_action()
				else:
					action = self.network(self.state_model(state)).detach().numpy()
					action = np.argmax(action)
				# 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
				next_state, reward, done = self.env.step(action)
				
				# reward = -10. if done else reward 	# 如果 Game Over，给予大的负奖励
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

		plot_loss_reward(loss_list, reward_list)


def plot_loss_reward(loss_list, reward_list):
	plt.subplot(1, 3, 1)
	plt.title('LOSS')
	plt.xlabel('episode')
	plt.ylabel('LOSS')
	plt.plot(range(len(loss_list)), loss_list)

	plt.subplot(1, 3, 3)
	plt.title('Total Reward')
	plt.xlabel('episode')
	plt.ylabel('Total Reward')
	plt.plot(range(len(reward_list)), reward_list)
	plt.show()


def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for Q Learing")
	# parser.add_argument("--render", type=bool, default=False)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument('--state_size', type=int, default=128)
	parser.add_argument('--hidden_size0', type=int, default=256)
	parser.add_argument('--hidden_size1', type=int, default=512)
	parser.add_argument('--output_size', type=int, default=4020)
	parser.add_argument('--gamma', type=float, default=0.95)
	parser.add_argument('--epoch', type=int, default=700)	# 610 个用户, 怎么也得 610 才能全部过一遍
	# 探索过程所占的episode数量
	parser.add_argument('--num_exploration_episodes', type=int, default=500)
	parser.add_argument('--initial_epsilon', type=float, default=1.0)	# 探索起始时的探索率
	parser.add_argument('--final_epsilon', type=float, default=0.01)	# 探索终止时的探索率
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--maxlen', type=int, default=1000)
	args = parser.parse_args()

	env = MovieLens_Env()

	q_learning = Q_Learning(args, env)
	q_learning.train()

if __name__ == '__main__':
	main()