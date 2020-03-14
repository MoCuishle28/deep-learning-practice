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
		self.seq_input_size = args.u_emb_dim + args.m_emb_dim + args.g_emb_dim + 1	# +1 是 rating
		self.hidden_size = args.seq_hidden_size
		self.seq_layer_num = args.seq_layer_num
		self.seq_output_size = args.seq_output_size

		# embedding layer
		self.u_embedding = nn.Embedding(args.max_uid + 1, args.u_emb_dim)
		self.m_embedding = nn.Embedding(args.max_mid + 1, args.m_emb_dim)
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
		uids = x[:, :, -(self.args.feature_size + 1)]
		mids = x[:, :, -((self.args.feature_size + 1) - 1)]
		genres = x[:, :, -((self.args.feature_size + 1) - 2):-1]
		rating = x[:, :, -1].view(x.shape[0], x.shape[1], -1)

		uemb = self.u_embedding(uids.long().to(self.device))
		memb = self.m_embedding(mids.long().to(self.device))
		gemb = self.g_embedding(genres.to(self.device))
		x = torch.cat([uemb, memb, gemb, rating], -1).to(self.device)

		h0 = torch.zeros(self.seq_layer_num, x.size(0), self.hidden_size).to(self.device)
		
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

		params = parse_layers(args.q_layers, self.activative_func, layer_trick, args.dropout, 1)
		self.q = nn.Sequential(*params)


	def forward(self, s, a):
		s = self.seq(s)
		x = torch.cat([s, a], dim=1)
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

		params = parse_layers(layers, self.activative_func, layer_trick, args.dropout, args.policy_output_size)
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
		self.alpha = nn.Parameter(torch.tensor([1], dtype=torch.float32).to(device))
		self.eta = nn.Parameter(torch.tensor([3], dtype=torch.float32).to(device))

		self.optim_dict = {'rms':torch.optim.RMSprop, 'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}

		self.prior_optim = self.optim_dict.get(args.optim, torch.optim.Adam)
		self.actor_optim = self.optim_dict.get(args.optim, torch.optim.Adam)
		self.critic_optim = self.optim_dict.get(args.optim, torch.optim.Adam)
		self.alpha_optim = self.optim_dict.get(args.optim, torch.optim.Adam)
		self.eta_optim = self.optim_dict.get(args.optim, torch.optim.Adam)

		self.prior_optim = self.prior_optim(self.prior.parameters(), lr=args.p_lr)
		self.actor_optim = self.actor_optim(self.policy.parameters(), lr=args.a_lr)
		self.critic_optim = self.critic_optim(self.Q.parameters(), lr=args.q_lr)
		self.alpha_optim = self.alpha_optim([self.alpha], lr=args.a_lr)
		self.eta_optim = self.eta_optim([self.eta], lr=args.q_lr)

		hard_update(self.target_Q, self.Q)
		# hard_update(self.target_policy, self.policy)
		self.Ntu = 0	# update target
		self.state_list, self.action_list, self.reward_list, self.next_state_list = [], [], [], []


	def select_action_from_target(self, state):
		output = self.target_policy(state).to(self.device)		# 输出 (batch, 2) 2 -> mu, log_std
		mu = output[:, 0]
		log_std = output[:, 1]
		z = torch.normal(mean=0, std=torch.ones(log_std.shape[-1])).to(self.device)
		a = mu + torch.exp(log_std) * z
		return a.clamp(min=self.args.min, max=self.args.max), mu, log_std


	def select_action_without_noise(self, state):
		output = self.policy(state).to(self.device)		# 输出 (batch, 2) 2 -> mu, log_std
		mu = output[:, 0]
		log_std = output[:, 1]
		a = mu + torch.exp(log_std)
		return a.clamp(min=self.args.min, max=self.args.max), mu, log_std


	def select_action(self, state):
		output = self.policy(state).to(self.device)		# 输出 (batch, 2) 2 -> mu, log_std
		mu = output[:, 0]
		log_std = output[:, 1]
		z = torch.normal(mean=0, std=torch.ones(log_std.shape[-1])).to(self.device)
		a = mu + torch.exp(log_std) * z
		return a.clamp(min=self.args.min, max=self.args.max), mu, log_std


	def select_action_from_prior(self, state):
		output = self.prior(state).to(self.device)		# 输出 (batch, 2) 2 -> mu, log_std
		mu = output[:, 0]
		log_std = output[:, 1]
		z = torch.normal(mean=0, std=torch.ones(log_std.shape[-1])).to(self.device)
		a = mu + torch.exp(log_std) * z
		return a.clamp(min=self.args.min, max=self.args.max), mu, log_std


	def gaussian_likelihood(self, x, mu, log_std):
		'''
		x: action (batch, feature size)
		return: log pi(a|s)
		'''
		# 运算变量必须都是 tensor (np.pi)
		pre_sum = -0.5 * ( ((x - mu) / (torch.exp(log_std) + EPS))**2 + 2 * log_std + torch.log(torch.tensor([2*np.pi]).to(self.device)) )
		return torch.sum(pre_sum, dim=-1)


	def optimize_model(self):
		gamma = self.args.gamma
		state_batch = torch.stack(self.state_list).to(self.device)
		# abm
		a, mu, log_std = self.select_action(state_batch)	# 维度只有 (batch)
		log_pi = self.gaussian_likelihood(a.unsqueeze(dim=1), mu.unsqueeze(dim=1), log_std.unsqueeze(dim=1))
		R = self._R().squeeze()
		values = self._values(state_batch).squeeze()
		abm_loss = -torch.sum(self._f(R - values) * log_pi)
		self.prior_optim.zero_grad()
		abm_loss.backward()
		self.prior_optim.step()

		reward_batch = torch.tensor(self.reward_list).view(len(self.reward_list), -1).to(self.device)
		action_batch = torch.tensor(self.action_list).view(len(self.action_list), -1).to(self.device)
		next_state_batch = torch.stack(self.next_state_list).to(self.device)
		# Q
		expected_next_values = self.args.gamma * self._values(next_state_batch).squeeze()
		curr_q_values = self.Q(state_batch, action_batch).to(self.device).squeeze()
		trajectory_len = len(self.state_list)
		q_loss = (1 / trajectory_len) * torch.sum((reward_batch.squeeze() + expected_next_values - curr_q_values)**2)
		self.critic_optim.zero_grad()
		q_loss.backward()
		self.critic_optim.step()
		# pi
		m_sum = 0
		eat_q_sum = 0
		for i in range(self.args.m):
			actions, mu, log_std = self.select_action_from_prior(state_batch)
			actions, mu, log_std = actions.unsqueeze(dim=1), mu.unsqueeze(dim=1), log_std.unsqueeze(dim=1)
			q_values = self.Q(state_batch, actions).to(self.device)
			log_prior = self.gaussian_likelihood(actions, mu, log_std).unsqueeze(1)

			_, mu, log_std = self.select_action(state_batch)
			mu, log_std = mu.unsqueeze(dim=1), log_std.unsqueeze(dim=1)
			log_pi = self.gaussian_likelihood(actions, mu, log_std).unsqueeze(1)
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

		eta_loss = -(self.eta * self.alpha + self.eta * torch.sum(torch.log(eat_q_sum)))
		self.eta_optim.zero_grad()
		eta_loss.backward()
		self.eta_optim.step()

		self.Ntu += 1
		if self.Ntu == self.args.update_period:
			self.Ntu = 0
			hard_update(self.target_Q, self.Q)
			# hard_update(self.target_policy, self.policy)

		self.clear_buffer()
		return a, abm_loss.item(), q_loss.item(), pi_loss.item(), alpha_loss.item(), eta_loss.item()


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
		actions = [self.select_action(state)[0].unsqueeze(dim=1) for _ in range(self.args.m)]
		values = torch.stack([self.target_Q(state, a).to(self.device).squeeze() for a in actions]).to(self.device)
		return (1 / self.args.m) * torch.sum(values, dim=0)


	def _R(self):
		# discounted R
		discounted_R = torch.zeros(len(self.state_list)).to(self.device)
		gamma = self.args.gamma
		sn = self.state_list[-1]
		sn = sn.view(1, sn.shape[0], sn.shape[1])
		sn_value = self._values(sn).to(self.device).squeeze()
		N = len(self.state_list)

		discounted_rewards = torch.zeros(len(self.reward_list)).to(self.device)
		running_add = 0
		# 相当于在计算 discounted return
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