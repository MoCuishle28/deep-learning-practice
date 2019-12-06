import pickle

import torch
from torch import nn
import numpy as np


def load_obj(name):
	with open('data/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)
		

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


epoch = 3
lr = 1e-3
plot_loss = []


movie_embedding = np.load('models/X_parameter_withoutNorm.npy')
user_embedding = np.load('models/Theta_parameter_withoutNorm.npy')
user_click_movieRow = load_obj('user_click_movieRow')

cfn = CFN(10, 10)

parameters = [cfn.uz, cfn.ui, cfn.wa, cfn.wi, cfn.wz, cfn.bz, cfn.bi]
optimizer = torch.optim.RMSprop(parameters, lr=lr)
loss_func = torch.nn.MSELoss()

for i in range(epoch):
	cnt = 0
	for uid, row_list in user_click_movieRow.items():
		target_state = torch.tensor(user_embedding[uid-1, :], dtype=torch.float32).view(10, 1)
		state = cfn.init_state()

		for row in row_list:
			cnt += 1
			action = torch.tensor(movie_embedding[row, :], dtype=torch.float32).view(10, 1)
			state = cfn(state, action)

		loss = loss_func(state, target_state)

		if cnt % 100 == 0:
			plot_loss.append(loss.item())
			print('Epoch:{}, cnt:{}, Loss:{}'.format(i, cnt, loss.item()))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


torch.save(cfn, 'models/cfn.pkl')

import matplotlib.pyplot as plt

plt.plot(plot_loss)
plt.show()