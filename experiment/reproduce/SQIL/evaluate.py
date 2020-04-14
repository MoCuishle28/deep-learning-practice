import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt


class Evaluation(object):
	def __init__(self, args, device, model, env):
		super(Evaluation, self).__init__()
		self.args = args
		self.device = device
		self.model = model
		self.env = env

		self.eval_data = np.load(args.base_data_dir + 'valid_data.npy').tolist() if args.mode == 'valid' else np.load(self.args.base_data_dir + 'test_data.npy').tolist()


	def eval(self):		
		hr_list, ndcg_list, precs_list = [], [], []
		for data in self.eval_data:
			uid, mid = data[0], data[1]
			state = self.env.get_history(uid, mid)
			state = state.view(1, state.shape[0], state.shape[1])
			action = self.model(state)
			_, rec_mids = action.topk(self.args.topk)
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