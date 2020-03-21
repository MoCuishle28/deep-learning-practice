import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


EPS = 1e-8


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


class SeqModel(nn.Module):
	def __init__(self, args, device):
		super(SeqModel, self).__init__()
		self.args = args
		self.device = device
		self.seq_input_size = args.u_emb_dim + args.m_emb_dim + args.g_emb_dim	# +1 是 rating
		self.hidden_size = args.seq_hidden_size
		self.seq_layer_num = args.seq_layer_num
		self.seq_output_size = args.seq_output_size

		# embedding layer
		self.u_embedding = nn.Embedding(args.max_uid + 1, args.u_emb_dim)
		self.m_embedding = nn.Embedding(args.max_mid + 1 + 1, args.m_emb_dim)	# 初始状态 mid=9742
		self.g_embedding = nn.Linear(args.feature_size - 2, args.g_emb_dim)
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.gru = nn.GRU(self.seq_input_size, self.hidden_size, self.seq_layer_num, batch_first=True)
		if args.norm_layer == 'bn':
			self.ln1 = nn.BatchNorm1d(self.hidden_size, affine=True)
		elif args.norm_layer == 'ln':
			self.ln1 = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
		self.fc = nn.Linear(self.hidden_size, self.seq_output_size)


	def forward(self, x):
		'''
		x: (batch, seq_len, feature_size)
		return: (batch, args.seq_output_size)
		'''
		uids = x[:, :, -self.args.feature_size]
		mids = x[:, :, -(self.args.feature_size - 1)]
		genres = x[:, :, -(self.args.feature_size - 2):]

		uemb = self.u_embedding(uids.long().to(self.device))
		memb = self.m_embedding(mids.long().to(self.device))
		gemb = self.g_embedding(genres.to(self.device))
		x = torch.cat([uemb, memb, gemb], -1).to(self.device)
		h0 = torch.zeros(self.seq_layer_num, x.size(0), self.hidden_size, device=self.device)

		out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
		out = out[:, -1, :]		# 最后时刻的 seq 作为输出
		if self.args.norm_layer != 'none':
			out = self.ln1(out)
		out = self.fc(out)
		return out


class Q(nn.Module):
	def __init__(self, args, seq, device):
		super(Q, self).__init__()
		self.args = args
		self.device = device
		self.seq = seq
		self.activative_func = get_activative_func(args.act)

		layer_trick = None
		if self.args.norm_layer == 'bn':
			layer_trick = nn.BatchNorm1d
		elif self.args.norm_layer == 'ln':
			layer_trick = nn.LayerNorm

		params = parse_layers(args.q_layers, self.activative_func, layer_trick, args.dropout, args.max_mid + 1)
		self.q = nn.Sequential(*params)


	def forward(self, s):
		x = self.seq(s)
		return self.q(x)


class Policy(nn.Module):
	def __init__(self, args, layers, seq, device):
		super(Policy, self).__init__()
		self.args = args
		self.device = device
		self.seq = seq
		self.activative_func = get_activative_func(args.act)

		layer_trick = None
		if self.args.norm_layer == 'bn':
			layer_trick = nn.BatchNorm1d
		elif self.args.norm_layer == 'ln':
			layer_trick = nn.LayerNorm

		params = parse_layers(layers, self.activative_func, layer_trick, args.dropout, args.max_mid + 1)
		self.model = nn.Sequential(*params)


	def forward(self, x):
		x = self.seq(x)
		return self.model(x)


class MPO(object):
	def __init__(self, args, device):
		super(MPO, self).__init__()
		self.args = args
		self.device = device

		self.target_seq = SeqModel(args, device).to(device)
		self.seq = SeqModel(args, device).to(device)
		# 和 policy 的运算, loss 不一样
		self.prior = Policy(args, args.p_layers, self.seq, device).to(device)
		
		self.target_Q = Q(args, self.target_seq, device).to(device)
		self.Q = Q(args, self.seq, device).to(device)

		# self.target_policy = Policy(args, args.a_layers, self.target_seq, device).to(device)
		self.policy = Policy(args, args.a_layers, self.seq, device).to(device)

		# 其他参数
		self.alpha = nn.Parameter(torch.tensor([1], dtype=torch.float32, device=device))
		self.eta = nn.Parameter(torch.tensor([3], dtype=torch.float32, device=device))

		self.optim_dict = {'rms':torch.optim.RMSprop, 'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}

		self.prior_optim = self.optim_dict.get(args.optim, torch.optim.Adam)
		self.actor_optim = self.optim_dict.get(args.optim, torch.optim.Adam)
		self.critic_optim = self.optim_dict.get(args.optim, torch.optim.Adam)
		self.alpha_optim = self.optim_dict.get(args.optim, torch.optim.Adam)
		self.eta_optim = self.optim_dict.get(args.optim, torch.optim.Adam)

		self.prior_optim = self.prior_optim(self.prior.parameters(), lr=args.p_lr, weight_decay=args.weight_decay)
		self.actor_optim = self.actor_optim(self.policy.parameters(), lr=args.a_lr, weight_decay=args.weight_decay)
		self.critic_optim = self.critic_optim(self.Q.parameters(), lr=args.q_lr, weight_decay=args.weight_decay)
		self.alpha_optim = self.alpha_optim([self.alpha], lr=args.a_lr, weight_decay=args.weight_decay)
		self.eta_optim = self.eta_optim([self.eta], lr=args.q_lr, weight_decay=args.weight_decay)

		hard_update(self.target_Q, self.Q)
		# hard_update(self.target_policy, self.policy)
		self.Ntu = 0	# update target
		self.state_list, self.action_list, self.reward_list, self.next_state_list = [], [], [], []


	def select_action_from_target_policy(self, state):
		output = self.target_policy(state).to(self.device)
		return torch.softmax(output, dim=1)


	def select_action(self, state):
		output = self.policy(state).to(self.device)
		return torch.softmax(output, dim=1)


	def select_action_from_prior(self, state):
		output = self.prior(state).to(self.device)		# 输出 (batch, args.max_mid)
		return torch.softmax(output, dim=1)


	def optimize_model(self):
		gamma = self.args.gamma
		state_batch = torch.stack(self.state_list).to(self.device)
		action_idx = torch.tensor(self.action_list, device=self.device).view(-1, 1)
		# abm
		softmax_abm_a = torch.softmax(self.prior(state_batch).to(self.device), dim=1)	# 维度只有 (batch)
		# (batch, 1)
		action_batch = torch.gather(softmax_abm_a, 1, action_idx).squeeze()

		R = self._R().squeeze()
		values = self._values(state_batch)
		abm_loss = -torch.sum(self._f(R - values) * torch.log(action_batch))
		self.prior_optim.zero_grad()
		abm_loss.backward()
		self.prior_optim.step()

		# (batch, 1)
		reward_batch = torch.tensor(self.reward_list, dtype=torch.float32, device=self.device).view(len(self.reward_list), -1)
		next_state_batch = torch.stack(self.next_state_list).to(self.device)

		# Q
		expected_next_values = self.args.gamma * self._values(next_state_batch).squeeze()
		q_values = self.Q(state_batch).to(self.device)
		curr_q_values = torch.gather(q_values, 1, action_idx)
		curr_q_values = curr_q_values.squeeze()

		q_loss = torch.mean((reward_batch.squeeze() + expected_next_values - curr_q_values)**2)
		self.critic_optim.zero_grad()
		q_loss.backward()
		self.critic_optim.step()

		m_sum = 0
		eat_q_sum = 0
		for i in range(self.args.m):
			softmax_abm_a = torch.softmax(self.prior(state_batch).to(self.device), dim=1)
			action_idx = softmax_abm_a.argmax(dim=1).unsqueeze(1)
			prior_action_batch = torch.gather(softmax_abm_a, 1, action_idx).squeeze()
			log_prior = torch.log(prior_action_batch)

			q_values = self.Q(state_batch).to(self.device)
			q_values = torch.gather(q_values, 1, action_idx).squeeze()

			softmax_pi_a = self.select_action(state_batch)
			pi_action_batch = torch.gather(softmax_pi_a, 1, action_idx).squeeze()
			log_pi = torch.log(pi_action_batch)

			m_sum = m_sum + (torch.exp(q_values / self.eta) * log_pi + self.alpha * (self.args.epsilon - F.kl_div(log_prior, torch.exp(log_pi), reduction='sum')))
			# eta loss part
			eat_q_sum = eat_q_sum + (torch.exp(q_values / self.eta))

		pi_loss = -torch.sum(m_sum)
		self.actor_optim.zero_grad()
		pi_loss.backward(retain_graph=True)
		self.actor_optim.step()

		alpha_loss = torch.sum(m_sum)
		self.alpha_optim.zero_grad()
		alpha_loss.backward(retain_graph=True)
		self.alpha_optim.step()

		eta_loss = -(self.eta * self.alpha + self.eta * torch.log(torch.sum(eat_q_sum / self.args.m)))
		self.eta_optim.zero_grad()
		eta_loss.backward()
		self.eta_optim.step()

		self.Ntu += 1
		if self.Ntu == self.args.update_period:
			self.Ntu = 0
			hard_update(self.target_Q, self.Q)
			# hard_update(self.target_policy, self.policy)

		action = softmax_pi_a.argmax(dim=1).to(self.device)
		target = torch.tensor(self.action_list, device=self.device).view(-1)
		precs = torch.mean((action == target).float())
		self.clear_buffer()
		return precs.item(), abm_loss.item(), q_loss.item(), pi_loss.item(), alpha_loss.item(), eta_loss.item()


	def _f(self, x):
		'''
		x: tensor
		return: if x >= 0 then _f(x) = 1 else _f(x) = 0
		'''
		return (x >= 0).float().to(self.device)


	def _values(self, state):
		'''
		state: batch state (batch, seq size, feature size)
		return values (batch, 1)
		'''
		batch_values = []
		for i in range(self.args.m):
			action_prob = self.select_action(state)
			q_values = self.target_Q(state)
			action_list = []
			for i_batch, prob in enumerate(action_prob.tolist()):
				dist = Categorical(torch.tensor(prob, device=self.device))
				action = dist.sample().item()
				action_list.append(action)

			values = torch.gather(q_values, 1, torch.tensor(action_list).view(-1, 1))
			batch_values.append(values.squeeze())
		batch_values = torch.stack(batch_values).to(self.device)
		return batch_values.mean(dim=0)


	def _R(self):
		# discounted R
		discounted_R = torch.zeros(len(self.state_list), device=self.device)
		gamma = self.args.gamma
		sn = self.state_list[-1]
		sn = sn.view(1, sn.shape[0], sn.shape[1])
		sn_value = self._values(sn)
		N = len(self.state_list)

		discounted_rewards = torch.zeros(len(self.reward_list), device=self.device)
		running_add = 0
		# 相当于在计算 discounted return, 最后一个是 0
		for t in reversed(range(0, len(self.reward_list) - 1)):
			running_add = running_add * gamma + self.reward_list[t]
			discounted_rewards[t] = running_add

		# t 从 1 开始
		running_add = 0
		for t in reversed(range(0, len(self.state_list))):
			running_add = running_add * gamma + sn_value
			discounted_R[t] = running_add
		discounted_R = discounted_R + discounted_rewards
		return discounted_R


	def store_transition(self, s, a, r):
		self.state_list.append(s)
		self.action_list.append(a)
		self.reward_list.append(r)

	def store_next_state(self, next_state):
		self.next_state_list.append(next_state)

	def clear_buffer(self):
		self.state_list.clear()
		self.action_list.clear()
		self.reward_list.clear()
		self.next_state_list.clear()

	def train(self):
		self.prior.train()
		# self.target_policy.train()
		self.policy.train()
		self.target_Q.train()
		self.Q.train()

	def eval(self):
		self.prior.eval()
		# self.target_policy.eval()
		self.policy.eval()
		self.target_Q.eval()
		self.Q.eval()