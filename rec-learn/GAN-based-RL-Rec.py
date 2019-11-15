import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


import time
import random


M = 5			# 只考虑 M 个历史记录
N = 10 			# 历史数据降维后的维度
K = 3			# k-rank 推荐	
eta = 1			# 参数 η
LR = 0.0001		# 学习率
EPOCH = 2		# epoch
dropout_probability1 = 0.8


def process_data():
	"""
	return: users, movies, movies_tags
	users(dict)-> 	userId:[ [uid, movieId 1, rating 1, time 1], [uid, movieId 2, rating 2, time 2]... ]
	movies(dict)-> 	movieId: [movieId, title, set(genres 1, genres 2)]
	movies_tags(list)-> tag 1, tag 2, ...
	users_have_saw(dict):	userId:{ movieId 1, movieId 2, ...}
	"""
	ratings_pd = pd.read_csv("ml-latest-small/ratings.csv", index_col=None)
	movies_pd = pd.read_csv("ml-latest-small/movies.csv", index_col=None)

	movies_tags = set()	# 电影标签
	for genres_list in movies_pd['genres']:
		movies_tags = movies_tags | set(genres_list.split('|'))
	movies_tags = list(movies_tags)		# 用作 one-hot
	movies_tags.remove('(no genres listed)') if '(no genres listed)' in movies_tags else _

	users = {}				# userId:[ [uid, movieId 1, rating 1, time 1], [uid, movieId 2, rating 2, time 2]... ]
	users_have_saw = {}		# userId:{ movieId 1, movieId 2, ...}
	ratings = ratings_pd.groupby(['userId'], as_index=False)

	for item in ratings:
		# item is tuple.  [0] is userId and [1] is data
		uid = item[0]
		users[uid] = list(zip(item[1]['userId'], item[1]['movieId'].values, item[1]['rating'].values, item[1]['timestamp'].values))
		users[uid] = list(map(list, users[item[0]]))
		users[uid] = [[int(x[0]), int(x[1]), float(x[2]), int(x[3])] for x in users[uid]]
		users_have_saw[uid] = { int(x[1]) for x in users[uid] }

	movies = {}		# movieId: [movieId, title, set(genres 1, genres 2) ]
	movies_pd = movies_pd.groupby(['movieId'], as_index=False)
	for item in movies_pd:
		movies[item[0]] = [item[1]['movieId'].values[0], item[1]['title'].values[0], set(item[1]['genres'].values[0].split('|'))]

	# 按照 timestamp 排序
	for k, v in users.items():
		users[k].sort(key=lambda pair:pair[-1])

	return users, movies, movies_tags, users_have_saw


def generate_data(users, movies, movies_tags):
	"""
	生产训练数据和测试数据
	"""
	# movieId: [feature 1, feature 2, ...]
	movies_feature_dict = { movieId:np.array([ 1 if tag in movies_data[-1] else 0 for tag in movies_tags]).reshape(len(movies_tags),1) for movieId, movies_data in movies.items() }
	
	# userId: [[movieId 1, feature 1, feature 2, ...]; [movieId 2, feature 1, feature 2, ...]; ...] -> matrix: shape = (dxm)
	user_history_dict = {}
	for uid, data in users.items():
		user_history_dict[uid] = np.array(movies_feature_dict[data[0][1]])
		for item in data[1:]:
			user_history_dict[uid] = np.concatenate((user_history_dict[uid], movies_feature_dict[item[1]]), axis=1)

	user_state_dict = {}				# uid: [[feature 1, feature 2, ...]; [feature 1, feature 2, ...]; ...]
	user_history_train_dict = {}		# 除去 M 个以外的历史记录作为训练集
	for uid, item_matrix in user_history_dict.items():
		user_state_dict[uid] = torch.tensor(item_matrix[:, :M], dtype=torch.float64).view((item_matrix.shape[0], M))
		user_history_train_dict[uid] = torch.tensor(item_matrix[:, M:], dtype=torch.float32)

	return user_history_dict, user_state_dict, user_history_train_dict, movies_feature_dict


class UserModel_pw(torch.nn.Module):
	def __init__(self, f_d, matrix_row, matrix_col, matrix_act_func, n_hidden, hidden_act_func, n_output):
		super().__init__()
		# f_d 是特征向量维度
		# matrix_row 等于考虑历史记录的条数 M, matrix_col 等于特征降维后的维度
		# n_input 等于将 state 和 item 拼接起来的维度 (n + d 维)
		self.matrix = torch.tensor(np.random.normal(0, 0.01, (matrix_row, matrix_col)), dtype=torch.float64)
		self.b = torch.zeros(1, dtype=torch.float64)
		self.matrix.requires_grad_(requires_grad=True)
		self.b.requires_grad_(requires_grad=True)

		act_func = {'relu':torch.relu, 'elu':torch.nn.functional.elu}
		self.matrix_act_func = act_func.get(matrix_act_func, torch.relu)

		self.f_layer = torch.nn.Linear(f_d, f_d)

		self.hidden = torch.nn.Linear(matrix_col * f_d + f_d, n_hidden)		
		self.hidden_act_func = act_func.get(hidden_act_func, torch.relu)
		self.ouput = torch.nn.Linear(n_hidden, n_output)


	def forward(self, state, item):
		state = self.matrix_act_func(torch.mm(state, self.matrix) + self.b)
		state = state.view((state.shape[0]*state.shape[1], -1))

		item = self.f_layer(item)

		state = state.float()
		in_feature = torch.cat([state, item.view((item.shape[1], -1))])
		
		in_feature = self.hidden_act_func(self.hidden(in_feature.view(-1, in_feature.shape[0])))
		in_feature = F.dropout(in_feature, p=dropout_probability1)
		reward = self.ouput(in_feature)
		return reward


def sample(uid, movies_feature_dict, users_have_saw):
	cnt = 0
	ret = []
	while cnt < K - 1:
		movieId = random.sample(movies_feature_dict.keys(), 1)[0]
		if movieId not in users_have_saw[uid]:
			ret.append(torch.tensor(movies_feature_dict[movieId], dtype=torch.float32))
			cnt += 1
	return ret


def train(user_model, users, users_have_saw, user_history_dict, user_state_dict, user_history_train_dict, movies_tags):
	plt.ion()	# 实时画图
	plt.show()
	loss_list = []

	optimizer = torch.optim.SGD(user_model.parameters(), lr=LR)
	for epoch_time in range(EPOCH):
		ii = 0
		epoch_plot = []	
		for uid, one_hot_arr in user_history_train_dict.items():
			for i in range(one_hot_arr.shape[1]):
				dis_exp_r_sum = 0
				r_true = 0

				true_click_one_hot = one_hot_arr[:, i].clone().view((-1, len(movies_tags)))
				state_one_hot_matrix = user_state_dict[uid]
				display_items_arr = sample(uid, movies_feature_dict, users_have_saw)	# list

				r_true = user_model(state_one_hot_matrix, true_click_one_hot)
				for item in display_items_arr:
					item = item.view((-1, len(movies_tags)))
					dis_exp_r_sum += torch.exp(user_model(state_one_hot_matrix, item))


				loss = GAN_loss(r_true, dis_exp_r_sum)
				optimizer.zero_grad()	# 先将梯度降为0
				loss.backward()			# 反向传递
				optimizer.step()		# 再用 optimizer 优化梯度	

				ii += 1
				loss_list.append(loss.data)
				epoch_plot.append(loss.data)
			print('No.', ii, sum([x.data for x in loss_list])/len(loss_list))
			loss_list.clear()

		plt.cla()
		plt.plot(np.array([it for it in range(len(epoch_plot))]), epoch_plot, 'r-', lw=1)
		plt.savefig("epoch"+str(epoch_time)+"loss.png")
		epoch_plot = []
		plt.pause(0.01)


def GAN_loss(r_true, dis_exp_r_sum):
	return (1/eta) * torch.log(dis_exp_r_sum + 1) - r_true



start = time.time()

users, movies, movies_tags, users_have_saw = process_data()
user_history_dict, user_state_dict, user_history_train_dict, movies_feature_dict = generate_data(users, movies, movies_tags)

user_model = UserModel_pw(len(movies_tags), M, N, 'elu', 256, 'elu', 1)

# item = user_history_train_dict[2][:, 1].clone().view((-1, len(movies_tags)))
# reward = user_model(user_state_dict[2], item)
# print(reward)

train(user_model, users, users_have_saw, user_history_dict, user_state_dict, user_history_train_dict, movies_tags)

print(time.time() - start)