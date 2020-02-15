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
	def __init__(self, args):
		super(SeqModel, self).__init__()
		self.seq_input_size = args.seq_input_size
		self.hidden_size = args.seq_hidden_size
		self.seq_layer_num = args.seq_layer_num
		self.seq_output_size = args.seq_output_size
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.gru = nn.GRU(self.seq_input_size, self.hidden_size, self.seq_layer_num, batch_first=True)
		self.fc = nn.Linear(self.hidden_size, self.seq_output_size)

	def forward(self, x):
		'''
		x: (batch, seq_len, feature_size)
		return: (batch, self.seq_output_size)
		'''
		h0 = torch.zeros(self.seq_layer_num, x.size(0), self.hidden_size)
		
		out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
		out = self.fc(out[:, -1, :])    # 最后时刻的 seq 作为输出
		return out
		

class Actor(nn.Module):
	def __init__(self, hidden_size, num_input, actor_output, seq_model):
		super(Actor, self).__init__()
		self.seq_model = seq_model

		# 也许可以试试把当前要推荐的 item feature 也考虑进去？(那就变成了输出关于user、item的embedding)
		self.linear1 = nn.Linear(num_input, hidden_size)
		self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=True)

		self.linear2 = nn.Linear(hidden_size, hidden_size*2)
		self.ln2 = nn.LayerNorm(hidden_size*2, elementwise_affine=True)

		self.mu = nn.Linear(hidden_size*2, actor_output)
		# why?
		# self.mu.weight.data.mul_(0.1)
		# self.mu.bias.data.mul_(0.1)

	def forward(self, inputs):
		x = inputs
		x = self.seq_model(inputs)

		x = self.linear1(x)
		x = self.ln1(x)
		x = F.relu(x)
		x = self.linear2(x)
		x = self.ln2(x)

		x = torch.tanh(x)
		mu = self.mu(x)
		return mu


class Critic(nn.Module):
	def __init__(self, hidden_size, num_input, actor_output, seq_model):
		super(Critic, self).__init__()
		self.seq_model = seq_model

		self.linear1 = nn.Linear(num_input + actor_output, hidden_size)
		self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=True)

		self.linear2 = nn.Linear(hidden_size, hidden_size*2)
		self.ln2 = nn.LayerNorm(hidden_size*2, elementwise_affine=True)

		self.V = nn.Linear(hidden_size*2, 1)
		# why?
		# self.V.weight.data.mul_(0.1)
		# self.V.bias.data.mul_(0.1)

	def forward(self, inputs, actions):
		x = inputs
		x = self.seq_model(inputs)

		x = torch.cat((x, actions), 1)
		x = self.linear1(x)
		x = self.ln1(x)
		x = F.relu(x)

		x = self.linear2(x)
		x = self.ln2(x)
		x = F.relu(x)
		V = self.V(x)
		return V


class DDPG(object):
	def __init__(self, args):
		self.seq_model = SeqModel(args)
		seq_params = [param for param in self.seq_model.parameters()]

		self.actor = Actor(args.hidden_size, args.seq_output_size, args.actor_output, self.seq_model)
		self.actor_target = Actor(args.hidden_size, args.seq_output_size, args.actor_output, self.seq_model)
		self.actor_perturbed = Actor(args.hidden_size, args.seq_output_size, args.actor_output, self.seq_model)
		actor_params = seq_params + [param for param in self.actor.parameters()]
		self.actor_optim = Adam(actor_params, lr=args.actor_lr)

		self.critic = Critic(args.hidden_size, args.seq_output_size, args.actor_output, self.seq_model)
		self.critic_target = Critic(args.hidden_size, args.seq_output_size, args.actor_output, self.seq_model)
		critic_params = seq_params + [param for param in self.critic.parameters()]
		self.critic_optim = Adam(critic_params, lr=args.critic_lr)

		self.gamma = args.gamma
		self.actor_tau = args.actor_tau
		self.critic_tau = args.critic_tau

		hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
		hard_update(self.critic_target, self.critic)


	def select_action(self, state, action_noise=None, param_noise=None):
		self.actor.eval()
		if param_noise is not None: 
			# 给参数加噪声的输出
			mu = self.actor_perturbed((Variable(state)))
		else:
			mu = self.actor((Variable(state)))

		self.actor.train()
		mu = mu.data

		if action_noise is not None:
			mu += torch.Tensor(action_noise.noise())
		return mu 		# 返回的是	torch.tensor


	def update_parameters(self, batch):
		state_batch = Variable(torch.cat(batch.state))
		action_batch = Variable(torch.cat(batch.action))
		reward_batch = Variable(torch.cat(batch.reward))
		mask_batch = Variable(torch.cat(batch.mask))
		next_state_batch = Variable(torch.cat(batch.next_state))
		
		next_action_batch = self.actor_target(next_state_batch)
		next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

		reward_batch = reward_batch.unsqueeze(1)
		mask_batch = mask_batch.unsqueeze(1)
		expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

		self.critic_optim.zero_grad()

		state_action_batch = self.critic((state_batch), (action_batch))

		value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
		value_loss.backward()
		self.critic_optim.step()

		self.actor_optim.zero_grad()

		# actor 要最大化 Q-value
		policy_loss = -self.critic((state_batch),self.actor((state_batch)))

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


	def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
		if not os.path.exists('models/'):
			os.makedirs('models/')

		if actor_path is None:
			actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix) 
		if critic_path is None:
			critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix)
			 
		print('Saving models to {} and {}'.format(actor_path, critic_path))
		torch.save(self.actor.state_dict(), actor_path)
		torch.save(self.critic.state_dict(), critic_path)


	def load_model(self, actor_path, critic_path):
		print('Loading models from {} and {}'.format(actor_path, critic_path))
		if actor_path is not None:
			self.actor.load_state_dict(torch.load(actor_path))
		if critic_path is not None: 
			self.critic.load_state_dict(torch.load(critic_path))