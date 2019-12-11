import pickle

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def load_obj(name):
	with open('data/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Net(nn.Module):
	def __init__(self, n_input, n_hidden, n_output):
		super().__init__()
		self.input_layer = nn.Linear(n_input, n_hidden)
		self.output_layer = nn.Linear(n_hidden, n_output)


	def choose_greedy(self, action, k):
		return action.topk(k)


	def forward(self, state):
		hidden = self.input_layer(state)
		output = self.output_layer(hidden)
		return torch.softmax(output, dim=-1)

		

class CFN(nn.Module):
	def __init__(self, n_state, n_action):
		super().__init__()
		self.uz = torch.tensor(np.random.normal(0.0, 0.35, (n_state, n_state)), 
			dtype=torch.float32)
		self.ui = torch.tensor(np.random.normal(0.0, 0.35, (n_state, n_state)), 
			dtype=torch.float32)
		self.wa = torch.tensor(np.random.normal(0.0, 0.35, (n_state, n_action)), 
			dtype=torch.float32)
		self.wi = torch.tensor(np.random.normal(0.0, 0.35, (n_state, n_action)), 
			dtype=torch.float32)
		self.wz = torch.tensor(np.random.normal(0.0, 0.35, (n_state, n_action)), 
			dtype=torch.float32)
		self.bz = torch.tensor(np.random.normal(0.0, 0.35, (1, 1)), 
			dtype=torch.float32)
		self.bi = torch.tensor(np.random.normal(0.0, 0.35, (1, 1)), 
			dtype=torch.float32)

		self.uz.requires_grad_(requires_grad=True)
		self.ui.requires_grad_(requires_grad=True)
		self.wa.requires_grad_(requires_grad=True)
		self.wi.requires_grad_(requires_grad=True)
		self.wz.requires_grad_(requires_grad=True)
		self.bz.requires_grad_(requires_grad=True)
		self.bi.requires_grad_(requires_grad=True)


	def init_state(self):
		return torch.tensor(np.zeros(10), dtype=torch.float32).view(10, 1)


	def forward(self, state, action):
		# state->(n_state, 1)  action->(n_action, 1)
		zt = torch.tanh(torch.mm(self.uz, state) + torch.mm(self.wz, action) + self.bz)
		it = torch.tanh(torch.mm(self.ui, state) + torch.mm(self.wi, action) + self.bi)
		next_state = (zt * torch.tanh(state)) + (it * torch.tanh(torch.mm(self.wa, action)))
		return next_state


movie_embedding = np.load('models/X_parameter_withoutNorm.npy')
cfn = CFN(10, 10)
behavior_policy = Net(10, 16, 9742)

action = movie_embedding[0, :]
action = torch.tensor(action.reshape(10, 1), dtype=torch.float32)
state = cfn.init_state()
# 10, 1    10, 1
# print(state.shape, action.shape)

next_state = cfn(state, action)

# print(next_state.shape)		# 10, 1
# print(next_state)


print('\n\n######################## softmax ############################')

batch_state = torch.stack([state.t(), next_state.t()], dim=0)

print('batch_state.shape:', batch_state.shape)

p = behavior_policy(batch_state)

print(p.shape)
print(p)
if len(p) > 1:
	print('sum:', torch.sum(p[0]).item(), ', ', torch.sum(p[1]).item())
	n,i = p[0].topk(1)
	print(n, i)

	n,i = p[1].topk(1)
	print(n, i)
else:
	n,i = p.topk(1)
	print(torch.sum(p))
	print(n, ' and ', i)


print('\n########################## train ##########################')

epoch = 2
lr = 1e-3
plot_loss = []

def train_softmax_cfn():
	optimizer = torch.optim.RMSprop([
					{'params': [cfn.uz, cfn.ui, cfn.wa, cfn.wi, cfn.wz, cfn.bz, cfn.bi]},
					{'params': behavior_policy.parameters()}
				], lr=lr)

	loss_func = torch.nn.CrossEntropyLoss()
	user_click_movieRow = load_obj('user_click_movieRow')
	cnt = 0

	for i in range(epoch):
		for uid, row_list in user_click_movieRow.items():
			movie = torch.tensor(movie_embedding[row_list[0], :], dtype=torch.float32).view(10, 1)
			state = cfn(cfn.init_state(), movie)
			for act_row in row_list[1:]:
				action = behavior_policy(state.t())
				movie = torch.tensor(movie_embedding[act_row, :], dtype=torch.float32).view(10, 1)
				next_state = cfn(state, movie)
				state = next_state
				loss = loss_func(action, torch.tensor(act_row).view(-1, ))

				cnt += 1
				if cnt%50 == 0:
					print('cnt:', cnt, 'Loss:', loss.item())
					plot_loss.append(loss.item())

				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()

	import matplotlib.pyplot as plt

	plt.plot(plot_loss)
	plt.show()

train_softmax_cfn()

def train_cfn():
	# 不收敛
	# torch.load('models/cfn.pkl')
	parameters = [cfn.uz, cfn.ui, cfn.wa, cfn.wi, cfn.wz, cfn.bz, cfn.bi]
	user_embedding = np.load('models/Theta_parameter_withoutNorm.npy')
	user_click_movieRow = load_obj('user_click_movieRow')
	print(user_embedding.shape)
	print(len(user_click_movieRow[1]))

	optimizer = torch.optim.RMSprop(parameters, lr=lr)
	loss_func = torch.nn.MSELoss()

	for i in range(epoch):
		cnt = 0
		for uid, row_list in user_click_movieRow.items():
			target_state = torch.tensor(user_embedding[uid-1, :], dtype=torch.float32).view(10, 1)
			state = cfn.init_state()
			cnt += 1

			for row in row_list:
				action = torch.tensor(movie_embedding[row, :], dtype=torch.float32).view(10, 1)
				next_state = cfn(state, action)
				state = next_state

			loss = loss_func(state, target_state)

			plot_loss.append(loss.item())
			print('Epoch:{}, cnt:{}, Loss:{}'.format(i, cnt, loss.item()))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	torch.save(cfn, 'models/cfn.pkl')

	import matplotlib.pyplot as plt

	plt.plot(plot_loss)
	plt.show()

# train_cfn()