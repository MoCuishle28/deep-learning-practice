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
# user_embedding = np.load('models/Theta_parameter_withoutNorm.npy')

user_click_movieRow = load_obj('user_click_movieRow')				# uid: [row 1, row 2, ...]
# user_has_clicked_movieRow = load_obj('user_has_clicked_movieRow')	# uid: {row 1, row 2, ...}


################################# 参数 #################################
K = 1	# top-K
batch = 64
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
		state = np.zeros(10).reshape(1, 10)		# state-> (1, 10)

		for row in click_row_list:
			reward = 1		# reward-> scalar
			# action-> scalar
			RL.store_transition(state, row, reward)
			choose_movie = movie_embedding[row, :]
			# TODO state 变化

			if RL.store_len() >= batch:
				RL.learn(behavior_policy)	# 传入 behavior_policy


if __name__ == '__main__':
	np.random.seed(RANDOMSEED)

	RL = PolicyGradient(n_actions = movie_num, n_features = observation_space, 
		n_hidden = n_hidden, learning_rate = 0.001, reward_decay = 0.95)

	behavior_policy = None
	state_model = None

	for i_episode in range(num_episodes):
		train(RL, behavior_policy, state_model)

	# torch.save(RL, 'models/pg_policy.pkl')