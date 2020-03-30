import sys
import os
from collections import namedtuple
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

"""
From: https://github.com/ikostrikov/pytorch-ddpg-naf
Changed a little
"""


Transition = namedtuple(
	'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class OUNoise:
	def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
		# np.random.seed(0)
		self.action_dimension = action_dimension
		self.scale = scale
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.state = np.ones(self.action_dimension) * self.mu
		self.reset()

	def reset(self):
		self.state = np.ones(self.action_dimension) * self.mu

	def noise(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
		self.state = x + dx
		return self.state * self.scale


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)


class SeqModel(nn.Module):
	def __init__(self, args, device):
		super(SeqModel, self).__init__()
		self.device = device
		self.seq_input_size = args.m_emb_dim + args.g_emb_dim
		self.hidden_size = args.seq_hidden_size
		self.seq_layer_num = args.seq_layer_num
		self.seq_output_size = args.seq_output_size

		# embedding layer
		self.u_embedding = nn.Embedding(args.max_uid + 1, args.u_emb_dim)
		self.m_embedding = nn.Embedding(args.max_mid + 1 + 1, args.m_emb_dim)	# 刚开始时 mid=9742
		self.g_embedding = nn.Linear(args.fm_feature_size - 2, args.g_emb_dim)
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.gru = nn.GRU(self.seq_input_size, self.hidden_size, self.seq_layer_num, batch_first=True)
		self.args = args
		if args.norm_layer == 'bn':
			self.ln1 = nn.BatchNorm1d(self.hidden_size, affine=True)
		elif args.norm_layer == 'ln':
			self.ln1 = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
		self.fc = nn.Linear(self.hidden_size + args.u_emb_dim, self.seq_output_size)


	def forward(self, x):
		'''
		x: (batch, seq_len, feature_size)
		return: (batch, args.seq_output_size)
		'''
		uids = x[:, 0, -self.args.fm_feature_size]
		mids = x[:, :, -(self.args.fm_feature_size - 1)]
		genres = x[:, :, -(self.args.fm_feature_size - 2):]

		uemb = self.u_embedding(uids.long().to(self.device))
		memb = self.m_embedding(mids.long().to(self.device))
		gemb = self.g_embedding(genres.to(self.device))
		x = torch.cat([memb, gemb], -1).to(self.device)

		h0 = torch.zeros(self.seq_layer_num, x.size(0), self.hidden_size, device=self.device)
		
		out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
		out = out[:, -1, :]		# 最后时刻的 seq 作为输出
		if self.args.norm_layer != 'none':
			out = self.ln1(out)
		out = torch.cat([uemb, out], -1).to(self.device)
		out = self.fc(out)
		return out
		

class Actor(nn.Module):
	def __init__(self, hidden_size, num_input, actor_output, seq_model, args):
		super(Actor, self).__init__()
		self.seq_model = seq_model
		self.args = args
		activative_func_dict = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leaky':nn.LeakyReLU(), 
		'selu':nn.SELU(), 'prelu':nn.PReLU(), 'tanh':nn.Tanh()}
		self.activative_func = activative_func_dict.get(args.a_act, nn.ReLU())

		# 也许可以试试把当前要推荐的 item feature 也考虑进去？(那就变成了输出关于user、item的embedding)
		self.linear1 = nn.Linear(num_input, hidden_size)

		self.linear2 = nn.Linear(hidden_size, hidden_size//2)

		if self.args.norm_layer == 'bn':
			self.ln1 = nn.BatchNorm1d(hidden_size, affine=True)
			self.ln2 = nn.BatchNorm1d(hidden_size//2, affine=True)
		elif self.args.norm_layer == 'ln':
			self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=True)
			self.ln2 = nn.LayerNorm(hidden_size//2, elementwise_affine=True)

		self.mu = nn.Linear(hidden_size//2, actor_output)


	def forward(self, inputs):
		x = inputs
		x = self.seq_model(inputs)

		x = self.linear1(x)
		x = self.ln1(x) if self.args.norm_layer != 'none' else x
		x = self.activative_func(x)
		x = self.linear2(x)
		x = self.ln2(x) if self.args.norm_layer != 'none' else x

		x = self.activative_func(x)
		mu = self.mu(x)
		return mu


class Critic(nn.Module):
	def __init__(self, hidden_size, num_input, actor_output, seq_model, args):
		super(Critic, self).__init__()
		self.seq_model = seq_model
		self.args = args
		activative_func_dict = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leaky':nn.LeakyReLU(), 
		'selu':nn.SELU(), 'prelu':nn.PReLU(), 'tanh':nn.Tanh()}
		self.activative_func = activative_func_dict.get(args.c_act, nn.ReLU())

		self.linear1 = nn.Linear(num_input + actor_output, hidden_size)

		self.linear2 = nn.Linear(hidden_size, hidden_size//2)

		if self.args.norm_layer == 'bn':
			self.ln1 = nn.BatchNorm1d(hidden_size, affine=True)
			self.ln2 = nn.BatchNorm1d(hidden_size//2, affine=True)
		elif self.args.norm_layer == 'ln':
			self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=True)
			self.ln2 = nn.LayerNorm(hidden_size//2, elementwise_affine=True)

		self.V = nn.Linear(hidden_size//2, 1)


	def forward(self, inputs, actions):
		x = inputs
		x = self.seq_model(inputs)

		x = torch.cat((x, actions), 1)
		x = self.linear1(x)
		x = self.ln1(x) if self.args.norm_layer != 'none' else x
		x = self.activative_func(x)

		x = self.linear2(x)
		x = self.ln2(x) if self.args.norm_layer != 'none' else x
		x = self.activative_func(x)
		V = self.V(x)
		return V


class DDPG(object):
	def __init__(self, args, device):
		self.device = device
		self.seq_model = SeqModel(args, self.device).to(self.device)
		self.target_seq_model = SeqModel(args, self.device).to(self.device)	# 还需要一个 seq_model 给 target network

		self.actor = Actor(args.hidden_size, args.seq_output_size, args.actor_output, self.seq_model, args).to(self.device)
		self.actor_target = Actor(args.hidden_size, args.seq_output_size, args.actor_output, self.target_seq_model, args).to(self.device)
		# self.actor_perturbed = Actor(args.hidden_size, args.seq_output_size, args.actor_output, self.target_seq_model, args).to(self.device)

		if args.actor_optim == 'adam':
			self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)
		elif args.actor_optim == 'sgd':
			self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=args.actor_lr, momentum=args.momentum)
		elif args.actor_optim == 'rmsprop':
			self.actor_optim = torch.optim.RMSprop(self.actor.parameters(), lr=args.actor_lr)

		self.critic = Critic(args.hidden_size, args.seq_output_size, args.actor_output, self.seq_model, args).to(self.device)
		self.critic_target = Critic(args.hidden_size, args.seq_output_size, args.actor_output, self.target_seq_model, args).to(self.device)

		if args.critic_optim == 'adam':
			self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)
		elif args.critic_optim == 'sgd':
			self.critic_optim = torch.optim.SGD(self.critic.parameters(), lr=args.critic_lr, momentum=args.momentum)
		elif args.critic_optim == 'rmsprop':
			self.critic_optim = torch.optim.RMSprop(self.critic.parameters(), lr=args.critic_lr)

		self.gamma = args.gamma
		self.actor_tau = args.actor_tau
		self.critic_tau = args.critic_tau

		hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
		hard_update(self.critic_target, self.critic)

	
	def on_train(self):
		self.actor.train()
		self.actor_target.train()
		# self.actor_perturbed.train()
		self.critic.train()
		self.critic_target.train()


	def on_eval(self):
		self.actor.eval()
		self.actor_target.eval()
		# self.actor_perturbed.eval()
		self.critic.eval()
		self.critic_target.eval()


	def select_action(self, state, action_noise=None, param_noise=None):
		self.actor.eval()
		if param_noise is not None: 
			# 给参数加噪声的输出
			mu = self.actor_perturbed((Variable(state)))
		else:
			mu = self.actor((Variable(state)))

		mu = mu.data

		if action_noise is not None:
			mu += torch.Tensor(action_noise.noise()).to(self.device)
		self.actor.train()
		return mu.to(self.device) 		# 返回的是	torch.tensor


	def update_parameters(self, batch):
		state_batch = Variable(torch.cat(batch.state).to(self.device))
		action_batch = Variable(torch.cat(batch.action).to(self.device))
		reward_batch = Variable(torch.cat(batch.reward).to(self.device))
		mask_batch = Variable(torch.cat(batch.mask).to(self.device))
		next_state_batch = Variable(torch.cat(batch.next_state).to(self.device))
		
		next_action_batch = self.actor_target(next_state_batch).to(self.device)
		next_state_action_values = self.critic_target(next_state_batch, next_action_batch).to(self.device)

		reward_batch = reward_batch.unsqueeze(1)
		mask_batch = mask_batch.unsqueeze(1)
		expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

		self.critic_optim.zero_grad()

		state_action_batch = self.critic((state_batch), (action_batch)).to(self.device)

		value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
		value_loss.backward()
		self.critic_optim.step()

		self.actor_optim.zero_grad()

		# actor 要最大化 Q-value
		policy_loss = -self.critic((state_batch),self.actor((state_batch)).to(self.device))

		policy_loss = policy_loss.mean()
		policy_loss.backward()
		self.actor_optim.step()

		soft_update(self.actor_target, self.actor, self.actor_tau)
		soft_update(self.critic_target, self.critic, self.critic_tau)
		return value_loss.item(), policy_loss.item()


	def perturb_actor_parameters(self, param_noise):
		"""Apply parameter noise to actor model, for exploration"""
		hard_update(self.actor_perturbed, self.actor)
		params = self.actor_perturbed.state_dict()
		for name in params:
			if 'ln' in name: 
				pass 
			param = params[name]
			param += torch.randn(param.shape) * param_noise.current_stddev


	def save_model(self, version, epoch):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		if not os.path.exists('models/' + version + '/'):
			os.makedirs('models/' + version + '/')

		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		torch.save(self.actor.state_dict(), based_dir + 'a_' + tail)
		torch.save(self.critic.state_dict(), based_dir + 'c_' + tail)
		torch.save(self.seq_model.state_dict(), based_dir + 's_' + tail)


	def load_model(self, version, epoch):
		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		self.actor.load_state_dict(torch.load(based_dir + 'a_' + tail))
		hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight

		self.critic.load_state_dict(torch.load(based_dir + 'c_' + tail))
		hard_update(self.critic_target, self.critic)

		self.seq_model.load_state_dict(torch.load(based_dir + 's_'+ tail))
		hard_update(self.target_seq_model, self.seq_model)