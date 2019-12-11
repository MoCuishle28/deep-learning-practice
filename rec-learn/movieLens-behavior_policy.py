import pickle

import torch
import torch.nn as nn
import numpy as np


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_obj(name):
	with open('data/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

		
	def forward(self, x):
		# Set initial hidden and cell states 
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		
		# Forward propagate LSTM
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

		# Decode the hidden state of the last time step (即: -1)
		out = self.fc(out[:, -1, :])
		return out


class Behavior(nn.Module):
	def __init__(self, n_input, n_hidden, n_output):
		super().__init__()
		self.input_layer = nn.Linear(n_input, n_hidden)
		self.output_layer = nn.Linear(n_hidden, n_output)


	def choose_greedy(self, state, k):
		return self(state).topk(k)


	def forward(self, state):
		hidden = self.input_layer(state)
		output = self.output_layer(hidden)
		return torch.softmax(output, dim=-1)


class MixModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size, n_input, n_hidden, n_output):
		super(MixModel, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

		self.input_layer = nn.Linear(n_input, n_hidden)
		self.output_layer = nn.Linear(n_hidden, n_output)

	def forward(self, x):
		# Set initial hidden and cell states 
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		
		# Forward propagate LSTM
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

		# Decode the hidden state of the last time step (即: -1)
		out = self.fc(out[:, -1, :])

		hidden = self.input_layer(out)
		output = self.output_layer(hidden)
		return output, out


def train1(lr, epoch, movie_embedding):
	model = MixModel(10, 32, 1, 10, 10, 32, 9742)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	loss_func = torch.nn.CrossEntropyLoss()
	user_click_movieRow = load_obj('user_click_movieRow')
	
	plot_loss = []
	for i in range(epoch):
		cnt = 0
		for uid, row_list in user_click_movieRow.items():
			state = np.zeros((1, 10))
			curr_state = [state]

			for act_row in row_list:
				input_data = torch.tensor(curr_state, dtype=torch.float32).reshape((1, len(curr_state), 10))
				action, next_state = model(input_data)
				state = next_state
				loss = loss_func(action, torch.tensor(act_row).view(-1, ))

				choose_movie = movie_embedding[act_row, :].reshape((1, 10))
				if len(curr_state) > 10:
					curr_state.pop(0)
				curr_state.append(choose_movie)

				cnt += 1
				if cnt%200 == 0:
					print("[100800/{}] Loss:{:.4f}".format(cnt, loss.item()))
					plot_loss.append(loss.item())

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

	torch.save(model.state_dict(), 'models/MixModel.ckpt')

	import matplotlib.pyplot as plt

	plt.plot(plot_loss)
	plt.xlabel('steps')
	plt.ylabel('Loss')
	plt.show()


def train0(lr, epoch, movie_embedding):
	state_model = RNN(10, 32, 1, 10)
	behavior_policy = Behavior(10, 32, 9742)

	optimizer = torch.optim.Adam([
					{'params': state_model.parameters()},
					{'params': behavior_policy.parameters()}
				], lr=lr)

	loss_func = torch.nn.CrossEntropyLoss()
	user_click_movieRow = load_obj('user_click_movieRow')

	# t = torch.ones(1, 10)
	# print(state_model(t.reshape((1, 1, 10))))
	# print(behavior_policy(t))
	# print('-----start-----')
	
	plot_loss = []
	for i in range(epoch):
		cnt = 0
		for uid, row_list in user_click_movieRow.items():
			state = torch.zeros(1, 10, dtype=torch.float32)
			curr_state = []

			for act_row in row_list:
				action = behavior_policy(state)
				choose_movie = movie_embedding[act_row, :]
				if len(curr_state) > 10:
					curr_state.pop(0)
				curr_state.append(choose_movie)
				input_data = torch.tensor(curr_state, dtype=torch.float32).reshape((1, len(curr_state), 10))
				next_state = state_model(input_data)
				state = next_state
				loss = loss_func(action, torch.tensor(act_row).view(-1, ))

				cnt += 1
				if cnt%200 == 0:
					print("100800/{}, Loss:{:.4f}".format(cnt, loss.item()))
					plot_loss.append(loss.item())

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# t = torch.ones(1, 10)
				# print('1.', state_model(t.reshape((1, 1, 10))))
				# print('2.', behavior_policy(t))
				# if cnt > 5:
				# 	assert 0>1

	torch.save(state_model.state_dict(), 'models/state_model.ckpt')
	torch.save(behavior_policy.state_dict(), 'models/behavior_policy.ckpt')

	import matplotlib.pyplot as plt

	plt.plot(plot_loss)
	plt.xlabel('steps')
	plt.ylabel('Loss')
	plt.show()


if __name__ == '__main__':
	lr = 1e-3
	epoch = 1
	movie_embedding = np.load('models/X_parameter_withoutNorm.npy')

	# train0(lr, epoch, movie_embedding)
	train1(lr, epoch, movie_embedding)