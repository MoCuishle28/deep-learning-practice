import pickle
import os
import time
import random

import numpy as np
import torch
from PG import PolicyGradient


def save_obj(obj, name):
	with open('data/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open('data/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

# (9742, 10)
movie_embedding = np.load('models/X_parameter_withoutNorm.npy')
# (611, 10)
user_embedding = np.load('models/Theta_parameter_withoutNorm.npy')

user_click_movieRow = load_obj('user_click_movieRow')				# uid: [row 1, row 2, ...]
user_has_clicked_movieRow = load_obj('user_has_clicked_movieRow')	# uid: {row 1, row 2, ...}

user_history_row = load_obj('user_history_row')		# uid: [row 1, row 2, ...] 作为 state 的一部分
user_state = {}				# uid: state
# TODO 用 RNN 求 state


################################# 参数 #################################
M = 5	# 最近 M 个历史选择作为 state 的一部分
K = 3	# top-K
batch = 64
observation_space = M*10 + 10	# history movie embedding (M*10) + user embedding (10) (之后改成RNN的)
movie_num = 9742
RANDOMSEED = 1  # random seed
num_episodes = 500
n_hidden = 16


def test():
	RL = torch.load('models/pg_policy.pkl')
	# TODO


def train(RL, behavior_policy):
	# train data -> (state, action, reward)
	for uid, click_row_list in user_click_movieRow.items():
		user = user_embedding[uid, :].reshape((1, 10))
		movie = movie_embedding[user_history_row[uid], :]
		state = np.vstack((movie, user))
		# state = torch.tensor(state.ravel(), dtype=torch.float32)

		history = set(user_history_row[uid])
		for row in click_row_list:
			action = [row]
			reward = 1
			for i in range(2):
				movieRow = random.randint(0, 9741)
				while movieRow in history:
					movieRow = random.randint(0, 9741)
				action.append(movieRow)
			for idx in action[1:]:
				if idx in user_has_clicked_movieRow[uid]:
					reward += 1
			# state-> tensor.Size([60])
			# action-> list len:3
			# reward-> scalars
			RL.store_transition(state, action, reward)

			if RL.store_len() >= batch:
				RL.learn(behavior_policy)	# 传入 behavior_policy


if __name__ == '__main__':
	np.random.seed(RANDOMSEED)

	RL = PolicyGradient(n_actions = movie_num, n_features = observation_space, 
		n_hidden = n_hidden, learning_rate = 0.001, reward_decay = 0.95)

	behavior_policy = None

	for i_episode in range(num_episodes):
		train(RL, behavior_policy)
	# torch.save(RL, 'models/pg_policy.pkl')