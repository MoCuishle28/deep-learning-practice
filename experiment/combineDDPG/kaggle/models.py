import os

import torch
import torch.nn as nn
import numpy as np


def parse_layers(layers, activative_func, layer_trick, p):
	params = []
	layers = [int(x) for x in layers.split(',')]
	for i, num in enumerate(layers[:-1]):
		params.append(nn.Linear(num, layers[i + 1]))
		if layer_trick != None:
			params.append(layer_trick(layers[i + 1]))
		params.append(activative_func)
		params.append(nn.Dropout(p=p))
	return params

def get_activative_func(key):
	activative_func_dict = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leaky':nn.LeakyReLU(), 
		'selu':nn.SELU(), 'prelu':nn.PReLU(), 'tanh':nn.Tanh()}
	return activative_func_dict.get(key, nn.ReLU())


class MLP(nn.Module):
	def __init__(self, args, device, without_rl=False):
		super(MLP, self).__init__()
		self.args = args
		self.device = device
		self.without_rl = without_rl

		self.activative_func = get_activative_func(args.mlp_act)
		if without_rl:
			self.i_embedding = nn.Embedding(args.max_iid + 1, args.i_emb_dim)

		layer_trick = None
		if self.args.layer_trick == 'bn':
			layer_trick = nn.BatchNorm1d
		elif self.args.layer_trick == 'ln':
			layer_trick = nn.LayerNorm
		params = parse_layers(args.mlp_layers, self.activative_func, layer_trick, args.dropout)
		self.mlp = nn.Sequential(*params)


	def forward(self, x):
		if self.without_rl:
			x = self.i_embedding(x.long().to(self.device))
			print(x.shape)	# DEBUG
			x = x.squeeze()
			print(x.shape)
			assert 0>1
		return self.mlp(x)


class Predictor(object):
	def __init__(self, args, model, device, without_rl=False):
		super(Predictor, self).__init__()
		self.args = args
		self.device = device
		self.model = model.to(self.device)
		self.without_rl = without_rl

		if args.predictor_optim == 'adam':
			self.optim = torch.optim.Adam(self.model.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)
		elif args.predictor_optim == 'sgd':
			self.optim = torch.optim.SGD(self.model.parameters(), lr=args.predictor_lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif args.predictor_optim == 'rms':
			self.optim = torch.optim.RMSprop(self.model.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)

		self.criterion = nn.CrossEntropyLoss()


	def predict(self, data):
		return self.model(data)


	def train(self, data, target):
		'''
		input: 	data->(batch, 1)  target->(batch)
		return: scalar, list (size:batch)
		'''
		prediction = self.model(data)	# (batch, max_iid + 1)
		loss = self.criterion(prediction, target)
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		if self.without_rl:
			rewards = None
		else:
			rewards = self.get_reward(prediction, target)
		return loss.item(), rewards


	def get_reward(self, prediction, target):
		rewards = []
		r = 0
		for x, y in zip(prediction, target):
			_, rec_list = x.topk(self.args.topk)
			rec_list = rec_list.tolist()
			if self.args.reward == 'ndcg':
				r = self.get_ndcg(rec_list, y)
			elif self.args.reward == '':
				# TODO
				pass
			rewards.append(r)
		return rewards

	def get_ndcg(self, rank_list, gt_item):
		for i, mid in enumerate(rank_list):
			if mid == gt_item:
				return (np.log(2.0) / np.log(i + 2.0)).item()
		return 0.0

	def on_train(self):
		self.model.train()

	def on_eval(self):
		self.model.eval()

	def save(self, version, epoch):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		if not os.path.exists('models/' + version + '/'):
			os.makedirs('models/' + version + '/')

		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		torch.save(self.model.cpu().state_dict(), based_dir + 'p_' + tail)

	def load(self, version, epoch):
		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		self.model.load_state_dict(torch.load(based_dir + 'p_' + tail).to(self.device))