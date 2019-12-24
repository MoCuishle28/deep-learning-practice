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


class MovieLens_Env(object):
	def __init__(self):
		self.state_model = StateModel(128, 256, 2, 128)
		self.movie_embedding_128_mini = load_obj('movie_embedding_128_mini')	# mid:embedding
		self.users_behavior = load_obj('users_rating')	# uid:[[mid, rating, timestamp], ...] 有序
		# uid:{mid:rating, ...}
		self.users_rating = {k:{x[0]:x[1] for x in v} for k,v in self.users_behavior.items()}

		self.uid_map_uRow = load_obj('uid_map_uRow')
		self.uRow_map_uid = {uRow:uid for uid, uRow in self.uid_map_uRow.items()}

		self.mid_map_mRow = load_obj('mid_map_mRow')
		self.mRow_map_mid = {mRow:mid for mid, mRow in self.mid_map_mRow.items()}

		self.curr_uRow = 0
		self.window = 3					# 考虑最近多少部电影
		self.curr_idx = self.window		# 进行到当前用户的第几时刻的电影了	(0~2 三部电影作为 history)
		self.curr_state = None			# 当前状态


	def reset(self):
		self.curr_uRow = 0
		uid = self.uRow_map_uid[self.curr_uRow]
		self.curr_state = self.init_state(uid)
		return self.curr_state


	def init_state(self, uid):
		history_list = self.users_behavior[uid][:self.window]	# [[mid, rating, timestamp], ...]
		mid_list = [item[0] for item in history_list]
		embedding_list = [self.movie_embedding_128_mini[mid] for mid in mid_list]
		input_data = torch.stack(embedding_list).reshape((1, self.window, 128))
		return self.state_model(input_data)


	def step(self, action):
		'''
		action: mRow (int)
		return: next_state, reward, done
		'''
		done = False
		uid = self.uid_map_uRow[self.curr_uRow]
		mid = self.mRow_map_mid[action]
		reward = self.get_reward(uid, mid)
		if reward != 0:
			# curr_state 要加入刚点击的 movie
			self.curr_idx +=1	# TODO 也不应该 +1 因为成功推荐的 item 不一定是下一个时刻的

		if self.curr_idx >= len(self.users_behavior[uid]):
			done = False
			self.curr_idx = self.window
			self.curr_uRow += 1
			uid = self.uRow_map_uid[self.curr_uRow]
			self.curr_state = self.init_state()
		return self.curr_state, reward, done


	def get_reward(self, uid, mid):
		return self.users_rating[uid][mid] if mid in self.users_rating else -1


def main():
	env = MovieLens_Env()
	print(env.reset())


if __name__ == '__main__':
	main()