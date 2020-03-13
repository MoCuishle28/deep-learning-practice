import pickle

import torch
import numpy as np


def save_obj(obj, base, name):
	with open(base + name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(base, name):
	with open(base + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class HistoryGenerator(object):
	def __init__(self, args, device):
		self.device = device
		self.args = args
		# mid: one-hot feature (21维 -> mid, genre, genre, ...)
		self.mid_map_mfeature = load_obj(args.base_data_dir, 'mid_map_mfeature')		
		self.users_rating = load_obj(args.base_data_dir, 'users_rating_without_timestamp') # uid:[[mid, rating], ...] 有序
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


	def get_user_all_behaviors(self, uid):
		return self.users_rating[uid].copy()