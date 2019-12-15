import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
import tensorlayer as tl


class PolicyGradient(nn.Module):

	def __init__(self, n_features, n_actions, n_hidden, learning_rate=0.001, reward_decay=0.95):
		super().__init__()
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []

		self.input_layer = nn.Linear(n_features, n_hidden)
		self.output_layer = nn.Linear(n_hidden, n_actions)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


	def forward(self, state):
		hidden = torch.relu(self.input_layer(state))
		out = self.output_layer(hidden)
		return torch.softmax(out, dim=1)



	def choose_action(self, s):
		"""
		choose action with probabilities.
		:param s: state
		:return: act
		"""
		_logits = self(torch.tensor(s, dtype=torch.float32))
		_probs = _logits.detach().numpy()
		return tl.rein.choice_action_by_probs(_probs.ravel())


	def choose_action_greedy(self, s, k):
		"""
		choose k action with greedy policy
		:param s: state -> shape (1, x)
		:param k: top k
		:return: n, idx  ->[[n 1, n 2, ...], [idx 1, idx 2, ...]]
		"""
		probs = self(torch.tensor(s, dtype=torch.float32))
		return probs.topk(k)	


	def store_transition(self, s, a, r):
		"""
		store data in memory buffer
		:param s: state 	# 每个元素要是 numpy
		:param a: act
		:param r: reward 	# 每个元素是 float 类型数字
		:return:
		"""
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)


	def store_len(self):
		return len(self.ep_rs)


	def learn(self, behavior_policy=None):
		"""
		off-policy update policy parameters
		:return: loss
		"""
		# discount and normalize episode reward (是每个时刻的 return，而不是reward)
		discounted_ep_rs_norm = torch.tensor(self._discount_and_norm_rewards(), dtype=torch.float32)

		# 每行是 feature vec (batch, n_actions)
		all_act_prob = self(torch.tensor(np.vstack(self.ep_obs), dtype=torch.float32))

		# 获轨迹动作的 one-hot
		ep_as = torch.LongTensor(self.ep_as).view(len(self.ep_as), 1)
		# shape -> (batch, n_actions)
		true_act_one_hot = torch.zeros(len(self.ep_as), self.n_actions).scatter_(dim=1, index=ep_as, value=1)
	
		if behavior_policy != None:
			behavior_prob = behavior_policy(torch.tensor(np.vstack(self.ep_obs), dtype=torch.float32))
			is_ratio = (all_act_prob * true_act_one_hot) / (behavior_prob * true_act_one_hot)
			neg_log_prob = torch.sum(is_ratio * -torch.log(all_act_prob) * true_act_one_hot, dim=1)
			print((behavior_prob * true_act_one_hot))	# 是0 导致 loss 为 nan
		else:
			neg_log_prob = torch.sum(-torch.log(all_act_prob) * true_act_one_hot, dim=1)


		# discounted_ep_rs_norm 是经过 discount 的 reward（实际上是return）
		loss = torch.sum(neg_log_prob * discounted_ep_rs_norm)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
		return loss.item()


	def _discount_and_norm_rewards(self):
		"""
		compute discount_and_norm_rewards
		:return: discount_and_norm_rewards
		"""
		# discount episode rewards
		discounted_ep_rs = np.zeros_like(self.ep_rs)	# 一个和 self.ep_rs 维度一样的全 0 ndarrage
		running_add = 0
		# 相当于在计算 return
		for t in reversed(range(0, len(self.ep_rs))):
			running_add = running_add * self.gamma + self.ep_rs[t]
			discounted_ep_rs[t] = running_add

		# normalize episode rewards
		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)
		return discounted_ep_rs