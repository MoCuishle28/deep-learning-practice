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
	with open('../data/ml_1M_row/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Evaluate(object):
	def __init__(self, args, device, agent, predictor, train_data, valid_data, test_data, env):
		self.args = args
		self.device = device
		self.agent = agent
		self.predictor = predictor
		self.train_data = train_data
		self.valid_data = valid_data
		self.test_data = test_data

		self.users_has_clicked = load_obj('users_has_clicked')	# uid:{mid, mid, ...}
		self.mid_map_mfeature = env.mid_map_mfeature
		self.env = env

		# self.build_ignore_set(train_data.tolist() + test_data.tolist())


	def get_hr(self, rank_list, gt_item):
		for mid in rank_list:
			if mid == gt_item:
				return 1
		return 0.0


	def get_ndcg(self, rank_list, gt_item):
		for i, mid in enumerate(rank_list):
			if mid == gt_item:
				return math.log(2.0, 2) / math.log(i + 2.0, 2)
		return 0.0


	def get_precs(self, rank_list, gt_item):
		for i, mid in enumerate(rank_list):
			if mid == gt_item:
				return 1.0 / (i + 1.0)
		return 0.0


	def get_current_action(self, uid, mid):
		state = self.env.get_history(uid, mid)
		state = state.view(-1, state.shape[0], state.shape[1]).to(self.device)
		action = self.agent.select_action(state, action_noise=None)
		return action


	def gen_predictor_input_data(self, uid, mid, action):
		uid_tensor = torch.tensor([uid], dtype=torch.float32, device=self.device)
		mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32, device=self.device)
		input_vector = torch.cat([action.squeeze(), uid_tensor, mfeature]).unsqueeze(0).to(self.device)	# 一维
		return input_vector


	def evaluate_for_user(self, args):
		uid, mid = args[0], args[1]
		ret = [0.0, 0.0, 0.0]	# hr, ndcg, precs
		map_items_score = {}	# mid: score

		action = self.get_current_action(uid, mid)
		input_vector = self.gen_predictor_input_data(uid, mid, action)
		max_score = self.predictor.predict(input_vector)

		# user_ignore_set = self.ignore_set[uid]
		user_ignore_set = self.users_has_clicked[uid]	# 忽略所有点击过的 item, 一次只评估一个 item

		count_larger = 0	# Early stop if there are args.topk items larger than max_score
		early_stop = False
		for i in range(self.args.max_mid + 1):
			if i in user_ignore_set:	# 忽略训练过的 item (以及在验证时忽略测试集,测试时忽略验证集)
				continue
			input_vector = self.gen_predictor_input_data(uid, i, action)
			score = self.predictor.predict(input_vector)
			map_items_score[i] = score
			if score > max_score:
				count_larger += 1
			if count_larger > self.args.topk:
				early_stop = True
				break

		if early_stop == False:
			rank_list = heapq.nlargest(self.args.topk, map_items_score, key=map_items_score.get)
			hr = self.get_hr(rank_list, mid)
			ndcg = self.get_ndcg(rank_list, mid)
			precs = self.get_precs(rank_list, mid)
			ret[0], ret[1], ret[2] = hr, ndcg, precs
		return ret


	def evaluate(self, title='[Valid]'):
		dataset = None
		self.hits, self.ndcgs, self.precs = [], [], []

		if title == '[Valid]':
			dataset = self.valid_data.tolist()
		else:	# Testing data set
			dataset = self.test_data.tolist()
			# self.build_ignore_set(self.train_data.tolist() + self.valid_data.tolist())
			print('Testing...')

		ret_list = []
		thread_func_args = [[vector[0], vector[1]] for vector in dataset]	# 0->uid, 1->mid
		if self.args.num_thread <= 1:	# 不用多线程
			for args in thread_func_args:
				ret = self.evaluate_for_user(args)
				ret_list.append(ret)
		else:
			pool = multiprocessing.Pool(processes=self.args.num_thread)
			ret_list = pool.map(self.evaluate_for_user, thread_func_args)
			pool.close()
			pool.join()

		for ret in ret_list:
			self.hits.append(ret[0])
			self.ndcgs.append(ret[1])
			self.precs.append(ret[2])

		self.hits = torch.tensor(self.hits, dtype=torch.float32).to(self.device)
		self.ndcgs = torch.tensor(self.ndcgs, dtype=torch.float32).to(self.device)
		self.precs = torch.tensor(self.precs, dtype=torch.float32).to(self.device)
		return self.hits.mean().item(), self.ndcgs.mean().item(), self.precs.mean().item()


	def build_ignore_set(self, ignore_list):
		# TODO 可是负采样的数据也是经历过训练的啊?
		self.ignore_set = {}	# uid: {mid, mid, ...}	用于 训练/测试 的 items, 在 验证集 评估时忽略 (测试集同理)
		for mfeature in ignore_list:
			uid = mfeature[0]
			mid = mfeature[1]

			if uid in self.ignore_set:
				self.ignore_set[uid].add(mid)
			else:
				self.ignore_set[uid] = set([mid])
		self.ignore_set


	def plot_result(self, args, loss_list, precision_list, hr_list, ndcg_list):
		plt.figure(figsize=(8, 8))
		plt.subplot(1, 7, 1)
		plt.title('Valid Precision')
		plt.xlabel('Step')
		plt.ylabel('Precision')
		plt.plot(precision_list)

		plt.subplot(1, 7, 3)
		plt.title('Valid HR')
		plt.xlabel('Step')
		plt.ylabel('HR')
		plt.plot(hr_list)

		plt.subplot(1, 7, 5)
		plt.title('Valid NDCG')
		plt.xlabel('Step')
		plt.ylabel('LOSS')
		plt.plot(ndcg_list)

		plt.subplot(1, 7, 7)
		plt.title('Training BPR LOSS')
		plt.xlabel('Step')
		plt.ylabel('LOSS')
		plt.plot(loss_list)

		plt.savefig(args.base_pic_dir + args.v + '.png')
		if args.show == 'y':
			plt.show()