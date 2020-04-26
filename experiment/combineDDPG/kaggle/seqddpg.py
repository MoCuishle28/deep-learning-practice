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

from seqmodel import SeqModel

"""
From: https://github.com/ikostrikov/pytorch-ddpg-naf
Changed a little
"""


Transition = namedtuple(
	'Transition', ('state', 'action', 'next_state', 'reward'))


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


def get_activative_func(key):
	activative_func_dict = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leaky':nn.LeakyReLU(), 
		'selu':nn.SELU(), 'prelu':nn.PReLU(), 'tanh':nn.Tanh()}
	return activative_func_dict.get(key, nn.ReLU())
		

class Actor(nn.Module):
	def __init__(self, seq_model, args):
		super(Actor, self).__init__()
		self.seq_model = seq_model
		self.args = args
		self.activative_func = get_activative_func(args.a_act)

		layer_trick = None
		if self.args.layer_trick == 'bn':
			layer_trick = nn.BatchNorm1d
		elif self.args.layer_trick == 'ln':
			layer_trick = nn.LayerNorm
		params = parse_layers(args.a_layers, self.activative_func, layer_trick, args.dropout, args.actor_output)
		self.actor = nn.Sequential(*params)

	def forward(self, inputs):
		x = inputs
		x = self.seq_model(inputs)
		return torch.tanh(self.actor(x))


class Critic(nn.Module):
	def __init__(self, seq_model, args):
		super(Critic, self).__init__()
		self.seq_model = seq_model
		self.args = args
		self.activative_func = get_activative_func(args.a_act)

		layer_trick = None
		if self.args.layer_trick == 'bn':
			layer_trick = nn.BatchNorm1d
		elif self.args.layer_trick == 'ln':
			layer_trick = nn.LayerNorm
		params = parse_layers(args.c_layers, self.activative_func, layer_trick, args.dropout, 1)
		self.critic = nn.Sequential(*params)

	def forward(self, inputs, actions):
		x = self.seq_model(inputs)
		x = torch.cat((x, actions), 1)
		return self.critic(x)


class DDPG(object):
	def __init__(self, args, device):
		self.device = device
		self.seq_model = SeqModel(args, self.device).to(self.device)
		self.target_seq_model = SeqModel(args, self.device).to(self.device)	# 还需要一个 seq_model 给 target network

		self.actor = Actor(self.seq_model, args).to(self.device)
		self.actor_target = Actor(self.target_seq_model, args).to(self.device)
		# self.actor_perturbed = Actor(self.target_seq_model, args).to(self.device)

		if args.actor_optim == 'adam':
			self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay)
		elif args.actor_optim == 'sgd':
			self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=args.actor_lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif args.actor_optim == 'rmsprop':
			self.actor_optim = torch.optim.RMSprop(self.actor.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay)

		self.critic = Critic(self.seq_model, args).to(self.device)
		self.critic_target = Critic(self.target_seq_model, args).to(self.device)

		if args.critic_optim == 'adam':
			self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
		elif args.critic_optim == 'sgd':
			self.critic_optim = torch.optim.SGD(self.critic.parameters(), lr=args.critic_lr, momentum=args.momentum, weight_decay=args.weight_decay)
		elif args.critic_optim == 'rmsprop':
			self.critic_optim = torch.optim.RMSprop(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)

		hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
		hard_update(self.critic_target, self.critic)

	
	def on_train(self):
		self.actor.train()
		self.critic.train()
		# self.actor_target.train()
		# self.actor_perturbed.train()
		# self.critic_target.train()


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
		# mask_batch = Variable(torch.cat(batch.mask).to(self.device))
		next_state_batch = Variable(torch.cat(batch.next_state).to(self.device))
		
		next_action_batch = self.actor_target(next_state_batch).detach().to(self.device)
		next_state_action_values = self.critic_target(next_state_batch, next_action_batch).detach().to(self.device)

		reward_batch = reward_batch.unsqueeze(1)
		# mask_batch = mask_batch.unsqueeze(1)
		# expected_state_action_batch = reward_batch + (self.args.gamma * mask_batch * next_state_action_values)
		expected_state_action_batch = reward_batch + (self.args.gamma * next_state_action_values)

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

		soft_update(self.actor_target, self.actor, self.args.actor_tau)
		soft_update(self.critic_target, self.critic, self.args.critic_tau)
		return policy_loss.item(), value_loss.item(), None


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
		torch.save(self.actor.cpu().state_dict(), based_dir + 'a_' + tail)
		torch.save(self.critic.cpu().state_dict(), based_dir + 'c_' + tail)


	def load_model(self, version, epoch):
		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'

		self.actor.load_state_dict(torch.load(based_dir + 'a_' + tail).to(self.device))
		hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight

		self.critic.load_state_dict(torch.load(based_dir + 'c_' + tail).to(self.device))
		hard_update(self.critic_target, self.critic)