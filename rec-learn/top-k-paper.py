import pickle
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from PG import PolicyGradient


def save_obj(obj, name):
	with open('data/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open('data/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class RNN(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

		
	def forward(self, x):
		# Set initial hidden and cell states 
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		
		# Forward propagate LSTM
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

		# Decode the hidden state of the last time step (即: -1)
		out = self.fc(out[:, -1, :])
		return out

class Behavior(torch.nn.Module):
	def __init__(self, n_input, n_hidden, n_output):
		super().__init__()
		self.input_layer = nn.Linear(n_input, n_hidden)
		self.output_layer = nn.Linear(n_hidden, n_output)

		init.xavier_normal_(self.input_layer.weight)
		init.xavier_normal_(self.output_layer.weight)


	def choose_greedy(self, state, k):
		return self(state).topk(k)


	def forward(self, state):
		hidden = torch.relu(self.input_layer(state))
		output = self.output_layer(hidden)
		return torch.softmax(output, dim=-1)	# dim=-1 例如：在有 batch 的时候针对最后一位 softmax


# (9742, 10)
movie_embedding = np.load('models/X_parameter_withoutNorm.npy')
# (611, 10)
# user_embedding = np.load('models/Theta_parameter_withoutNorm.npy')

user_click_movieRow = load_obj('user_click_movieRow')				# uid: [row 1, row 2, ...]
# user_has_clicked_movieRow = load_obj('user_has_clicked_movieRow')	# uid: {row 1, row 2, ...}


################################# 参数 #################################
K = 1	# top-K
batch = 128
observation_space = 10
movie_num = 9742
num_episodes = 1
n_hidden = 32


def test():
	RL = torch.load('models/pg_policy.pkl')
	# TODO


def train(RL, behavior_policy, state_model):
	# train data -> (state, action, reward)
	for uid, click_row_list in user_click_movieRow.items():
		state = torch.zeros(1, 10, dtype=torch.float32)
		curr_state = [np.zeros(10) for _ in range(5)]

		for row in click_row_list:
			reward = 1.0		# reward-> scalar
			# action-> scalar
			RL.store_transition(state, row, reward)

			choose_movie = movie_embedding[row, :]
			curr_state.pop(0)
			curr_state.append(choose_movie)

			input_data = torch.tensor(curr_state, dtype=torch.float32).reshape((1, len(curr_state), 10))
			next_state = state_model(input_data).detach().numpy()
			state = next_state

			if RL.store_len() >= batch:
				loss = RL.learn(behavior_policy)	# 传入 behavior_policy
				print('UID:{} Loss:{:.4f}'.format(uid, loss))


if __name__ == '__main__':
	RL = PolicyGradient(n_actions = movie_num, n_features = observation_space, 
		n_hidden = n_hidden, learning_rate = 0.001, reward_decay = 0.95)

	state_model = RNN(10, 16, 4, 10)
	state_model.load_state_dict(torch.load('models/state_model.ckpt'))	# 直接加载模型
	behavior_policy = Behavior(10, 32, 9742)
	behavior_policy.load_state_dict(torch.load('models/behavior_policy.ckpt'))	# 直接加载模型

	for i_episode in range(num_episodes):
		train(RL, behavior_policy, state_model)

	# torch.save(RL, 'models/pg_policy.pkl')