import pickle
import heapq
import multiprocessing

import torch
import numpy as np
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')


class Evaluate(object):
	def __init__(self, args, device, predictor, users_has_clicked, mid_map_mfeature):
		self.args = args
		self.device = device
		self.predictor = predictor 		# 只是 model, 不是封装的 predictor

		self.users_has_clicked = users_has_clicked
		self.mid_map_mfeature = mid_map_mfeature

		self.mid_dir = 'without_time_seq/' if args.without_time_seq == 'y' else ''
		self.valid_data = np.load(args.base_data_dir + self.mid_dir + 'valid_data.npy')


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


	def gen_predictor_input_data(self, uid, mid):
		uid = torch.tensor([uid], dtype=torch.float32, device=self.device)
		mfeature = torch.tensor(self.mid_map_mfeature[mid].astype(np.float32), dtype=torch.float32, device=self.device)
		input_vector = torch.cat([uid, mfeature])
		return input_vector.view(1, -1)


	def evaluate_for_user(self, args):
		uid, mid = args[0], args[1]
		ret = [0.0, 0.0, 0.0]	# hr, ndcg, precs
		map_items_score = {}	# mid: score

		input_vector = self.gen_predictor_input_data(uid, mid)
		max_score = self.predictor(input_vector)
		map_items_score[mid] = max_score.item()
		user_ignore_set = self.users_has_clicked[uid]	# 忽略所有点击过的 item, 一次只评估一个 item

		count_larger = 0	# Early stop if there are args.topk items larger than max_score
		early_stop = False
		for i in range(self.args.max_mid + 1):
			if i in user_ignore_set:	# 忽略训练过的 item (以及在验证时忽略测试集,测试时忽略验证集)
				continue
			input_vector = self.gen_predictor_input_data(uid, i)
			score = self.predictor(input_vector)
			map_items_score[i] = score.item()
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
			del self.valid_data
			test_data = np.load(self.args.base_data_dir + self.mid_dir + 'test_data.npy')
			dataset = test_data.tolist()
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

		self.hits = torch.tensor(self.hits, dtype=torch.float32, device=self.device)
		self.ndcgs = torch.tensor(self.ndcgs, dtype=torch.float32, device=self.device)
		self.precs = torch.tensor(self.precs, dtype=torch.float32, device=self.device)
		return self.hits.mean().item(), self.ndcgs.mean().item(), self.precs.mean().item()


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