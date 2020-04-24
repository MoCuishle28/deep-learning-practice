import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Evaluation(object):
	def __init__(self, args, device, agent):
		super(Evaluation, self).__init__()
		self.args = args
		self.device = device
		self.agent = agent
		
		if args.mode == 'valid':
			self.eval_sessions = pd.read_pickle(args.base_data_dir + 'sampled_val.df')
		elif args.mode == 'test':
			self.eval_sessions = pd.read_pickle(args.base_data_dir + 'sampled_test.df')


	def compute_index(self, state_batch, action_batch):
		state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
		prediction_batch = self.agent(state_batch)
		for prediction, action in zip(prediction_batch, action_batch):
			_, rec_iids = prediction.topk(self.args.topk)
			rec_iids = rec_iids.view(-1).tolist()
			hr, ndcg, precs = self.get_hr(rec_iids, action), self.get_ndcg(rec_iids, action), self.get_precs(rec_iids, action)
			self.hr_list.append(hr), self.ndcg_list.append(ndcg), self.precs_list.append(precs)


	def eval(self):
		eval_ids = self.eval_sessions.session_id.unique()
		groups = self.eval_sessions.groupby('session_id')
		self.hr_list, self.ndcg_list, self.precs_list = [], [], []
		state_batch, action_batch = [], []
		for sid in eval_ids:
			group = groups.get_group(sid)
			history = []
			for index, row in group.iterrows():
				if history == []:
					history.append(row['item_id'])
					continue
				state = list(history)
				state = self.pad_history(state, 10, self.args.max_iid)
				state_batch.append(state)
				action_batch.append(row['item_id'])
				history.append(row['item_id'])

			if len(state_batch) >= self.args.batch_size:
				self.compute_index(state_batch, action_batch)
				state_batch, action_batch = [], []
		
		if len(state_batch)	!= 0:
			self.compute_index(state_batch, action_batch)
		hr = torch.tensor(self.hr_list, dtype=torch.float32, device=self.device)
		ndcg = torch.tensor(self.ndcg_list, dtype=torch.float32, device=self.device)
		precs = torch.tensor(self.precs_list, dtype=torch.float32, device=self.device)
		return hr.mean().item(), ndcg.mean().item(), precs.mean().item()


	def pad_history(self, itemlist, length, pad_item):
		if len(itemlist) >= length:
			return itemlist[-length:]
		if len(itemlist) < length:
			temp = [pad_item] * (length - len(itemlist))
			itemlist.extend(temp)
			return itemlist


	def get_hr(self, rank_list, gt_item):
		for iid in rank_list:
			if iid == gt_item:
				return 1
		return 0.0

	def get_ndcg(self, rank_list, gt_item):
		for i, iid in enumerate(rank_list):
			if iid == gt_item:
				return (np.log(2.0) / np.log(i + 2.0)).item()
		return 0.0

	def get_precs(self, rank_list, gt_item):
		for i, iid in enumerate(rank_list):
			if iid == gt_item:
				return 1.0 / (i + 1.0)
		return 0.0

	def plot_result(self, precision_list, hr_list, ndcg_list):
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