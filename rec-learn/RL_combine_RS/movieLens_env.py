import pickle
import argparse
import random

import torch
import torch.nn as nn
import numpy as np


def save_obj(obj, name):
	with open('../data/ml20/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open('../data/ml20/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class MovieLens_Env(object):
	def __init__(self):
		self.movie_embedding_128_mini = load_obj('movie_embedding_128_mini')	# mid:embedding
		self.users_behavior = load_obj('users_rating')	# uid:[[mid, rating, timestamp], ...] 有序
		# uid:{mid:rating, ...}
		self.users_rating = {k:{x[0]:x[1] for x in v} for k,v in self.users_behavior.items()}

		self.uid_map_uRow = load_obj('uid_map_uRow')
		self.uRow_map_uid = {uRow:uid for uid, uRow in self.uid_map_uRow.items()}

		self.mid_map_mRow = load_obj('mid_map_mRow')
		self.mRow_map_mid = {mRow:mid for mid, mRow in self.mid_map_mRow.items()}

		self.curr_uRow = random.randint(0, 610)
		self.window = 5					# 考虑最近多少部电影
		self.curr_state = None			# state 是多个 embedding 叠成的 tensor
		self.negative_cnt = 0			# 负面推荐 terminal_negative_cnt 次则接受当前 episode
		self.terminal_negative_cnt = 3


	def reset(self):
		# 返回的 state 是多个 embedding 叠成的 tensor
		# self.curr_uRow = 0
		uid = self.uRow_map_uid[self.curr_uRow]
		self.curr_state = self.init_state(uid)
		return self.curr_state


	def init_state(self, uid):
		history_list = self.users_behavior[uid][:self.window]	# [[mid, rating, timestamp], ...]
		mid_list = [item[0] for item in history_list]
		embedding_list = [self.movie_embedding_128_mini[mid] for mid in mid_list]
		if len(embedding_list) < self.window:
			embedding_list = list(embedding_list)
			for i in range(self.window - len(embedding_list)):
				embedding_list.append(torch.zeros(128, dtype=torch.float32))
		input_data = torch.stack(embedding_list).reshape((1, len(embedding_list), 128))
		return input_data


	def step(self, action):
		'''
		action: mRow (int)
		return: next_state, reward, done
		'''
		done = False
		uid = self.uRow_map_uid[self.curr_uRow]
		mid = self.mRow_map_mid[action]
		reward = self.get_reward(uid, mid)
		# 没有评分算负面推荐, 或者 < 3 算负面
		self.negative_cnt = self.negative_cnt + 1 if reward <= 0 else self.negative_cnt

		# 更新 state
		movie_embedding = self.movie_embedding_128_mini[mid]
		movie_embedding = movie_embedding.reshape((1, movie_embedding.shape[0]))
		self.curr_state = self.curr_state[:, 1:, :].reshape((self.curr_state[:, 1:, :].shape[1], self.curr_state[:, 1:, :].shape[2]))
		self.curr_state = torch.cat([self.curr_state, movie_embedding], dim=0).reshape(1, self.window, 128)

		if self.negative_cnt > self.terminal_negative_cnt:
			self.negative_cnt = 0
			done = True
			self.curr_uRow = self.curr_uRow + 1 if self.curr_uRow + 1 in self.uRow_map_uid else 0
			uid = self.uRow_map_uid[self.curr_uRow]
			self.curr_state = self.init_state(uid)
		return self.curr_state, reward, done


	def get_reward(self, uid, mid):
		# 若评分过则返回 rating 否则 -1
		return self.users_rating[uid].get(mid, 0)


	def sample_action(self):
		return random.randint(0, 4019)


def test_env(env):
	for i in range(1220):
		state = env.reset()
		while True:
			action = env.sample_action()
			next_state, reward, done = env.step(action)
			state = next_state

			print(state.shape, reward, done, env.curr_uRow)
			if done:
				break

def main():
	env = MovieLens_Env()
	test_env(env)


if __name__ == '__main__':
	main()