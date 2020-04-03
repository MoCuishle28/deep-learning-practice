import os
import random
from collections import deque

import torch
import torch.nn as nn
from torch.distributions import Categorical


def parse_layers(layers, activative_func, layer_trick, p, output_size):
	params = []
	layers = [int(x) for x in layers.split(',')]
	for i, num in enumerate(layers[:-1]):
		params.append(nn.Linear(num, layers[i + 1]))
		if layer_trick != None:
			params.append(layer_trick(layers[i + 1]))
		params.append(activative_func)
		params.append(nn.Dropout(p=p))
	params.append(nn.Linear(layers[-1], output_size))
	return params


def get_activative_func(act):
	activative_func_dict = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leaky':nn.LeakyReLU(), 
		'selu':nn.SELU(), 'prelu':nn.PReLU(), 'tanh':nn.Tanh()}
	return activative_func_dict.get(act, nn.ReLU())


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)


class QNet(nn.Module):
	def __init__(self, args, device):
		super(QNet, self).__init__()
		self.args = args
		self.device = device
		activative_func = get_activative_func(args.act)

		# embedding 用于处理当前 positive data 和 uid
		self.u_embedding = nn.Embedding(args.max_uid + 1, args.u_emb_dim)
		self.m_embedding = nn.Embedding(args.max_mid + 1, args.m_emb_dim)
		self.g_embedding = nn.Linear(args.feature_size - 2, args.g_emb_dim)
		# fc 作为第一层输入处理
		self.fc = nn.Linear(args.u_emb_dim + args.m_emb_dim + args.g_emb_dim + args.k*2, int(args.layers.split(',')[0]))

		layer_trick = None
		if self.args.layer_trick == 'bn':
			layer_trick = nn.BatchNorm1d
		elif self.args.layer_trick == 'ln':
			layer_trick = nn.LayerNorm

		params = parse_layers(args.layers, activative_func, layer_trick, args.dropout, args.max_mid + 1)
		self.model = nn.Sequential(*params)
	

	def forward(self, x):
		uids = x[:, 0]
		mids = x[:, 1]
		genres = x[:, 2:self.args.feature_size]
		params = x[:, self.args.feature_size:]

		uemb = self.u_embedding(uids.long().to(self.device))
		memb = self.m_embedding(mids.long().to(self.device))
		gemb = self.g_embedding(genres.to(self.device))
		x = torch.cat([uemb, memb, gemb, params], dim=1).to(self.device)
		x = self.fc(x)
		return self.model(x)


class Q_Sampler(object):
	'''
	state-> 		1.正样本特征向量(uid, mid, gerne, ...)  2.模型参数 作为 state
	action-> 		负样本作为 action
	next state-> 	1.正样本/负样本, 2.更新后的参数作为 next state
	reward->		负的 y_ij(希望找到更难区分的负样本,对抗的思想), 
						或离散化的 y_ij(y_ij < 0 then reward = 1 else reward = -1)
	'''
	def __init__(self, args, device, users_has_clicked):
		super(Q_Sampler, self).__init__()
		self.args = args
		self.device = device
		self.replay_buffer = deque(maxlen=args.maxlen)
		self.lossFunc = nn.MSELoss()
		# uid:[mid, mid, ...]	用户的正样本
		self.user_pos = {uid:list(midset) for uid, midset in users_has_clicked.items()}
		
		self.Q = QNet(args, device).to(device)
		self.target_Q = QNet(args, device).to(device)
		hard_update(self.target_Q, self.Q)
		self.ntu = 0

		self.optim = None
		if args.q_optim == 'sgd':
			self.optim = torch.optim.SGD(self.Q.parameters(), lr=args.q_lr, weight_decay=args.weight_decay, momentum=args.momentum)
		elif args.q_optim == 'adam':
			self.optim = torch.optim.Adam(self.Q.parameters(), lr=args.q_lr, weight_decay=args.weight_decay)
		elif args.q_optim == 'rms':
			self.optim = torch.optim.RMSprop(self.Q.parameters(), lr=args.q_lr, weight_decay=args.weight_decay)


	def argmax_sample(self, uid, state):
		# 只针对一个 user
		q_values = self.target_Q(state)
		pos_items = self.user_pos[uid]
		q_values[0, pos_items] = 0
		return q_values.argmax().item()


	def softmax_sample(self, uid, state):
		# 只针对一个 user
		q_values = self.target_Q(state)
		pos_items = self.user_pos[uid]
		q_values[0, pos_items] = 0
		prob = torch.softmax(q_values, dim=-1)
		dist = Categorical(torch.tensor(prob, device=self.device))
		return dist.sample().item()


	def train(self):
		self.on_train()
		batch_state, batch_action, batch_reward, batch_next_state = zip(
						*random.sample(self.replay_buffer, self.args.batch_size))

		batch_state = torch.stack(batch_state)
		batch_action = torch.stack(batch_action)
		batch_reward = torch.stack(batch_reward)
		batch_next_state = torch.stack(batch_next_state)

		next_q_values = self.target_Q(batch_next_state)
		next_prediction = batch_reward.squeeze() + (self.args.gamma * torch.max(next_q_values, dim=1).values)

		current_q_values = self.Q(batch_state)
		one_hot_act = torch.zeros(self.args.batch_size, self.args.max_mid + 1, device=self.device).scatter_(dim=1, index=batch_action.long(), value=1)
		current_prediction = torch.sum(current_q_values * one_hot_act, dim=-1)

		loss = self.lossFunc(next_prediction, current_prediction)
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		if self.args.update_method == 'hard':
			self.ntu += 1
			if self.ntu == self.args.ntu:
				self.ntu = 0
				hard_update(self.target_Q, self.Q)
		else:
			soft_update(self.target_Q, self.Q, self.args.tau)

		self.on_eval()
		return loss.item()


	def on_eval(self):
		self.Q.eval()

	def on_train(self):
		self.Q.train()

	def save(self, version, epoch):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		if not os.path.exists('models/' + version + '/'):
			os.makedirs('models/' + version + '/')

		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		torch.save(self.Q.state_dict(), based_dir + 'Q_' + tail)


	def load(self, version, epoch):
		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		self.Q.load_state_dict(torch.load(based_dir + 'Q_'+ tail))
		hard_update(self.target_Q, self.Q)