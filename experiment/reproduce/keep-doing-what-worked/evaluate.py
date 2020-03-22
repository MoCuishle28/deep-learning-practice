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


	def evaluate(self, title='[Valid]'):
		if title == '[TEST]':
			data = np.load(self.args.base_data_dir + 'seq_predict/' + 'test_data.npy').tolist()
			target = np.load(self.args.base_data_dir + 'seq_predict/' + 'test_target.npy').tolist()
			self.build_dataset(data, target)

		result = []
		for uid, target in self.user_action.items():
			for action in target:
				state = self.env.get_history(uid, action)
				state = state.view(1, state.shape[0], state.shape[1])
				pred_prob = self.agent.select_action(state)
				pred_action = pred_prob.argmax(dim=1)
				result.append(action == pred_action)
		return torch.tensor(result, dtype=torch.float32, device=self.device).mean().item()


	def plot_result(self, args, train_precision_list, valid_precision_list):
		plt.figure(figsize=(8, 8))
		plt.subplot(1, 3, 1)
		plt.title('Training Precision')
		plt.xlabel('Step')
		plt.ylabel('Precision')
		plt.plot(train_precision_list)

		plt.subplot(1, 3, 3)
		plt.title('Valid Precision')
		plt.xlabel('Step')
		plt.ylabel('Precision')
		plt.plot(valid_precision_list)

		plt.savefig(args.base_pic_dir + args.v + '.png')
		if args.show == 'y':
			plt.show()