import pickle
import time
import random
import math
import heapq
import multiprocessing

import torch
import numpy as np
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')


def load_obj(name):
	with open('../../data/ml_1M_row/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Evaluate(object):
	def __init__(self, args, device, agent, env):
		self.args = args
		self.device = device
		self.agent = agent

		self.mid_map_mfeature = env.mid_map_mfeature
		self.env = env

		valid_data = np.load(args.base_data_dir + 'seq_predict/' + 'valid_data.npy').tolist()
		valid_target = np.load(args.base_data_dir + 'seq_predict/' + 'valid_target.npy').tolist()
		self.build_dataset(valid_data, valid_target)


	def build_dataset(self, data, target):
		self.user_action = {uid:[] for uid in range(1, self.args.max_uid + 1)}
		for pair, prediction in zip(data, target):
			uid = pair[0]
			self.user_action[uid].append(prediction)


	def evaluate(self, mode='valid'):
		if mode == 'test':
			data = np.load(self.args.base_data_dir + 'seq_predict/' + 'test_data.npy').tolist()
			target = np.load(self.args.base_data_dir + 'seq_predict/' + 'test_target.npy').tolist()
			self.build_dataset(data, target)

		hr_list, ndcg_list, precs_list = [], [], []
		for uid, target in self.user_action.items():
			for mid in target:
				state = self.env.get_history(uid, mid)
				state = state.view(1, state.shape[0], state.shape[1])
				scores = self.agent.policy(state)
				_, rec_mids = scores.topk(self.args.topk)
				rec_mids = rec_mids.view(-1).tolist()

				hr, ndcg, precs = self.get_hr(rec_mids, mid), self.get_ndcg(rec_mids, mid), self.get_precs(rec_mids, mid)
				hr_list.append(hr)
				ndcg_list.append(ndcg)
				precs_list.append(precs)

		hr = torch.tensor(hr_list, dtype=torch.float32, device=self.device)
		ndcg = torch.tensor(ndcg_list, dtype=torch.float32, device=self.device)
		precs = torch.tensor(precs_list, dtype=torch.float32, device=self.device)
		return hr.mean().item(), ndcg.mean().item(), precs.mean().item()


	def get_hr(self, rank_list, gt_item):
		for mid in rank_list:
			if mid == gt_item:
				return 1
		return 0.0


	def get_ndcg(self, rank_list, gt_item):
		for i, mid in enumerate(rank_list):
			if mid == gt_item:
				return (np.log(2.0) / np.log(i + 2.0)).item()
		return 0.0


	def get_precs(self, rank_list, gt_item):
		for i, mid in enumerate(rank_list):
			if mid == gt_item:
				return 1.0 / (i + 1.0)
		return 0.0


	def plot_result(self, hr_list, ndcg_list, precision_list):
		plt.figure(figsize=(8, 8))
		plt.subplot(1, 5, 1)
		plt.title('Valid Precision')
		plt.xlabel('Step')
		plt.ylabel('Precision')
		plt.plot(precision_list)

		plt.subplot(1, 5, 3)
		plt.title('Valid HR')
		plt.xlabel('Step')
		plt.ylabel('HR')
		plt.plot(hr_list)

		plt.subplot(1, 5, 5)
		plt.title('Valid NDCG')
		plt.xlabel('Step')
		plt.ylabel('LOSS')
		plt.plot(ndcg_list)

		plt.savefig(self.args.base_pic_dir + self.args.v + '.png')
		if self.args.show == 'y':
			plt.show()