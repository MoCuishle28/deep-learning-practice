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
		return F.softmax(out, dim=1)



	def choose_action(self, s):
		"""
		choose action with probabilities.
		:param s: state
		:return: act
		"""
		_logits = self(torch.tensor(s, torch.float32))
		_probs = self(_logits).detach().numpy()
		return tl.rein.choice_action_by_probs(_probs.ravel())


	def choose_action_greedy(self, s, k):
		"""
		choose k action with greedy policy
		:param s: state
		:return: act
		"""
		_probs = torch.softmax(self.model(torch.tensor(s, dtype=torch.float32))).detach().numpy()
		# return _probs[np.argpartition(_probs, -k)[-k:]]		# top-k
		return _probs.topk(k)	# 返回 [[n 1, n 2, ...], [idx 1, idx 2, ...]]


	def store_transition(self, s, a, r):
		"""
		store data in memory buffer
		:param s: state
		:param a: act
		:param r: reward
		:return:
		"""
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)


	def store_len(self):
		return len(self.rs)


	def learn(self, behavior_policy):
		"""
		off-policy update policy parameters
		:return: None
		"""
		return
		# TODO
		# discount and normalize episode reward (是每个时刻的 return，而不是reward)
		discounted_ep_rs_norm = self._discount_and_norm_rewards()

		# 每行是 feature vec    列是这个 batch 有几个 vec
		all_act_prob = self(torch.tensor(np.vstack(self.ep_obs)))

		# torch没有 one-hot	TODO
		true_act_one_hot = tf.one_hot(self.ep_as, self.n_actions)
		true_act_prob = all_act_prob * true_act_one_hot

		neg_log_prob = tf.reduce_sum(-torch.log(all_act_prob)*true_act_one_hot, axis=1)

		# discounted_ep_rs_norm 是经过 discount 的 reward（实际上是return）
		loss = tf.reduce_mean(neg_log_prob * discounted_ep_rs_norm)  # reward(实际上是 return) guided loss

		grad = tape.gradient(loss, self.model.trainable_weights)
		self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

		self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
		return discounted_ep_rs_norm


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