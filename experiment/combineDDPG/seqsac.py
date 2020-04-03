import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable

# test
import argparse
import matplotlib.pyplot as plt
import gym


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)


def get_act_func(key):
	activative_func_dict = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leaky':nn.LeakyReLU(), 
		'selu':nn.SELU(), 'prelu':nn.PReLU(), 'tanh':nn.Tanh()}
	return activative_func_dict.get(key, nn.ReLU())


class SeqModel(nn.Module):
	def __init__(self, args, device):
		super(SeqModel, self).__init__()
		self.args = args
		self.device = device
		self.seq_input_size = args.m_emb_dim + args.g_emb_dim	# 加一维特征表示是否点击 TODO
		self.hidden_size = args.seq_hidden_size
		self.seq_layer_num = args.seq_layer_num

		# embedding layer
		self.u_embedding = nn.Embedding(args.max_uid + 1, args.u_emb_dim)
		self.m_embedding = nn.Embedding(args.max_mid + 1 + 1, args.m_emb_dim)	# 刚开始时 mid=9742
		self.g_embedding = nn.Linear(args.fm_feature_size - 2, args.g_emb_dim)
		# 加一维特征表示是否点击 TODO

		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.gru = nn.GRU(self.seq_input_size, self.hidden_size, self.seq_layer_num, batch_first=True)
		if args.norm_layer == 'bn':
			self.ln1 = nn.BatchNorm1d(self.hidden_size, affine=True)
		elif args.norm_layer == 'ln':
			self.ln1 = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

		self.fc = nn.Linear(self.hidden_size + args.u_emb_dim, args.seq_output_size)


	def forward(self, x):
		'''
		x: (batch, seq_len, feature_size)
		return: (batch, args.seq_output_size)
		'''
		uids = x[:, 0, -self.args.fm_feature_size]
		mids = x[:, :, -(self.args.fm_feature_size - 1)]
		genres = x[:, :, -(self.args.fm_feature_size - 2):]
		# 加一维特征表示是否点击 TODO

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

# SAC
class ValueNetwork(nn.Module):
	def __init__(self, args, device, seq_model):
		super(ValueNetwork, self).__init__()
		self.args = args
		self.device = device
		self.seq_model = seq_model
		state_dim = args.seq_output_size
		hidden_1 = args.hidden_size
		hidden_2 = hidden_1 // 2
		self.activative_func = get_act_func(args.v_act)

		self.linear1 = nn.Linear(state_dim, hidden_1)
		self.linear2 = nn.Linear(hidden_1, hidden_2)
		self.linear3 = nn.Linear(hidden_2, 1)

		if self.args.norm_layer == 'bn':
			self.ln1 = nn.BatchNorm1d(hidden_1, affine=True)
			self.ln2 = nn.BatchNorm1d(hidden_2, affine=True)
		elif self.args.norm_layer == 'ln':
			self.ln1 = nn.LayerNorm(hidden_1, elementwise_affine=True)
			self.ln2 = nn.LayerNorm(hidden_2, elementwise_affine=True)
		else:
			self.ln1, self.ln2 = None, None
		

	def forward(self, state):
		state = self.seq_model(state)
		x = self.linear1(state)
		x = self.ln1(x) if self.ln1 != None else x
		x = self.activative_func(x)
		x = self.linear2(x)
		x = self.ln2(x) if self.ln2 != None else x
		x = self.activative_func(x)
		x = self.linear3(x)
		return x
		
		
class SoftQNetwork(nn.Module):
	def __init__(self, args, device, seq_model):
		super(SoftQNetwork, self).__init__()
		self.args = args
		self.device = device
		self.seq_model = seq_model
		num_inputs, num_actions, hidden_1 = args.seq_output_size, args.actor_output, args.hidden_size
		hidden_2 = hidden_1//2
		self.activative_func = get_act_func(args.c_act)

		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_1)
		self.linear2 = nn.Linear(hidden_1, hidden_2)
		self.linear3 = nn.Linear(hidden_2, 1)

		if self.args.norm_layer == 'bn':
			self.ln1 = nn.BatchNorm1d(hidden_1, affine=True)
			self.ln2 = nn.BatchNorm1d(hidden_2, affine=True)
		elif self.args.norm_layer == 'ln':
			self.ln1 = nn.LayerNorm(hidden_1, elementwise_affine=True)
			self.ln2 = nn.LayerNorm(hidden_2, elementwise_affine=True)
		else:
			self.ln1, self.ln2 = None, None
		

	def forward(self, state, action):
		state = self.seq_model(state)
		x = torch.cat([state, action], 1)
		x = self.linear1(x)
		x = self.ln1(x) if self.ln1 != None else x
		x = self.activative_func(x)
		x = self.linear2(x)
		x = self.ln2(x) if self.ln2 != None else x
		x = self.activative_func(x)
		x = self.linear3(x)
		return x
		
		
class PolicyNetwork(nn.Module):
	def __init__(self, args, device, seq_model):
		super(PolicyNetwork, self).__init__()
		self.args = args
		self.device = device
		self.seq_model = seq_model
		num_inputs, num_actions, hidden_1 = args.seq_output_size, args.actor_output, args.hidden_size
		hidden_2 = hidden_1 // 2
		self.activative_func = get_act_func(args.a_act)

		self.log_std_min = args.log_std_min
		self.log_std_max = args.log_std_max
		
		self.linear1 = nn.Linear(num_inputs, hidden_1)
		self.linear2 = nn.Linear(hidden_1, hidden_2)

		if self.args.norm_layer == 'bn':
			self.ln1 = nn.BatchNorm1d(hidden_1, affine=True)
			self.ln2 = nn.BatchNorm1d(hidden_2, affine=True)
		elif self.args.norm_layer == 'ln':
			self.ln1 = nn.LayerNorm(hidden_1, elementwise_affine=True)
			self.ln2 = nn.LayerNorm(hidden_2, elementwise_affine=True)
		else:
			self.ln1, self.ln2 = None, None
		
		self.mean_linear = nn.Linear(hidden_2, num_actions)
		self.log_std_linear = nn.Linear(hidden_2, num_actions)
		

	def forward(self, state):
		state = self.seq_model(state)
		x = self.linear1(state)
		x = self.ln1(x) if self.ln1 != None else x
		x = self.activative_func(x)
		x = self.linear2(x)
		x = self.ln2(x) if self.ln2 != None else x
		x = self.activative_func(x)
		
		mean    = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		return mean, log_std
	

	def evaluate(self, state, epsilon=1e-8):
		mean, log_std = self.forward(state)
		std = log_std.exp()
		
		normal = Normal(mean, std)
		z = normal.sample()
		action = torch.tanh(z)
		
		log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
		log_prob = log_prob.sum(-1, keepdim=True)
		
		return action, log_prob, z, mean, log_std
		
	
	def get_action(self, state):
		mean, log_std = self.forward(state)
		std = log_std.exp()
		
		normal = Normal(mean, std)
		z = normal.sample()
		action = torch.tanh(z)
		return action


class SAC(object):
	def __init__(self, args, device):
		super(SAC, self).__init__()
		self.args = args
		self.device = device

		self.seq_model = SeqModel(args, self.device).to(self.device)
		self.target_seq_model = SeqModel(args, self.device).to(self.device)

		self.value_net = ValueNetwork(args, device, self.seq_model).to(device)
		self.target_value_net = ValueNetwork(args, device, self.target_seq_model).to(device)

		self.soft_q_net = SoftQNetwork(args, device, self.seq_model).to(device)
		self.policy_net = PolicyNetwork(args, device, self.seq_model).to(device)

		self.value_criterion  = nn.MSELoss()
		self.soft_q_criterion = nn.MSELoss()

		self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.value_lr)
		self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=args.critic_lr)
		self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.actor_lr)

		hard_update(self.target_value_net, self.value_net)


	def update_parameters(self, batch):
		state = Variable(torch.cat(batch.state).to(self.device))
		action = Variable(torch.cat(batch.action).to(self.device))
		reward = Variable(torch.cat(batch.reward).unsqueeze(1).to(self.device))		# 注意要扩展一维
		next_state = Variable(torch.cat(batch.next_state).to(self.device))

		expected_q_value = self.soft_q_net(state, action)
		expected_value = self.value_net(state)
		new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

		target_value = self.target_value_net(next_state)
		next_q_value = reward + self.args.gamma * target_value

		q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

		expected_new_q_value = self.soft_q_net(state, new_action)
		next_value = expected_new_q_value - log_prob
		value_loss = self.value_criterion(expected_value, next_value.detach())

		log_prob_target = expected_new_q_value - expected_value
		policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
		

		mean_loss = self.args.mean_lambda * mean.pow(2).mean()
		std_loss = self.args.std_lambda * log_std.pow(2).mean()
		z_loss = self.args.z_lambda * z.pow(2).sum(1).mean()

		policy_loss += mean_loss + std_loss + z_loss

		self.soft_q_optimizer.zero_grad()
		q_value_loss.backward()
		self.soft_q_optimizer.step()

		self.value_optimizer.zero_grad()
		value_loss.backward()
		self.value_optimizer.step()

		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		self.policy_optimizer.step()
		soft_update(self.target_value_net, self.value_net, self.args.critic_tau)
		return q_value_loss.item(), policy_loss.item(), value_loss.item()


	def on_eval(self):
		self.value_net.eval()
		self.target_value_net.eval()
		self.soft_q_net.eval()
		self.policy_net.eval()


	def on_train(self):
		self.value_net.train()
		self.target_value_net.train()
		self.soft_q_net.train()
		self.policy_net.train()


	def save_model(self, version, epoch):
		if not os.path.exists('models/'):
			os.makedirs('models/')
		if not os.path.exists('models/' + version + '/'):
			os.makedirs('models/' + version + '/')

		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		torch.save(self.value_net.state_dict(), based_dir + 'value_' + tail)
		torch.save(self.soft_q_net.state_dict(), based_dir + 'soft_q_' + tail)
		torch.save(self.policy_net.state_dict(), based_dir + 'policy_' + tail)


	def load_model(self, version, epoch):
		based_dir = 'models/' + version + '/'
		tail = version + '-' + str(epoch) + '.pkl'
		self.value_net.load_state_dict(torch.load(based_dir + 'value_' + tail))
		hard_update(self.target_value, self.value_net)

		self.soft_q_net.load_state_dict(torch.load(based_dir + 'soft_q_' + tail))
		self.policy_net.load_state_dict(torch.load(based_dir + 'policy_'+ tail))