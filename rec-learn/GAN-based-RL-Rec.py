import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging


import time
import random


M = 10			# 只考虑 M 个历史记录
N = 5 			# 历史数据乘以矩阵后输出的矩阵 col
K = 3			# k-rank 推荐	
eta = 1			# 参数 η
LR = 0.0001		# 学习率
EPOCH = 5		# epoch
dropout_probability1 = 0.5
dropout_probability2 = 0.5
n_embedding = 5	# 降维后


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
		user_state_dict[uid] = torch.tensor(item_matrix[:, :M], dtype=torch.float32).view((item_matrix.shape[0], M))
		user_history_train_dict[uid] = torch.tensor(item_matrix[:, M:], dtype=torch.float32)

	return user_history_dict, user_state_dict, user_history_train_dict, movies_feature_dict


class UserModel_pw(torch.nn.Module):
	def __init__(self, f_d, matrix_row, matrix_col, matrix_act_func, n_hidden_1, hidden_act_func_1, n_hidden_2,hidden_act_func_2, n_output):
		super().__init__()
		# f_d 是特征向量维度
		# matrix_row 等于考虑历史记录的条数 M, matrix_col 等于特征降维后的维度
		# n_input 等于将 state 和 item 拼接起来的维度 (n + d 维)

		self.embedding = torch.nn.Linear(f_d, n_embedding)

		self.matrix = torch.nn.Linear(matrix_row, matrix_col)

		act_func = {'relu':torch.relu, 'elu':torch.nn.functional.elu}
		self.matrix_act_func = act_func.get(matrix_act_func, torch.relu)


		self.hidden_1 = torch.nn.Linear(matrix_col * n_embedding + n_embedding, n_hidden_1)		
		self.hidden_act_func_1 = act_func.get(hidden_act_func_1, torch.relu)

		self.hidden_2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
		self.hidden_act_func_2 = act_func.get(hidden_act_func_2, torch.relu)

		self.ouput = torch.nn.Linear(n_hidden_2, n_output)


	def forward(self, state, item):
		state = self.embedding(state.t())
		state = state.t()
		item = self.embedding(item)

		state = self.matrix_act_func(self.matrix(state))
		state = state.view((state.shape[0]*state.shape[1], -1))

		state = state.float()
		in_feature = torch.cat((state, item.view((item.shape[1], -1))))
		
		in_feature = self.hidden_act_func_1(self.hidden_1(in_feature.view(-1, in_feature.shape[0])))
		in_feature = F.dropout(in_feature, p = dropout_probability1)

		in_feature = self.hidden_act_func_2(self.hidden_2(in_feature))
		in_feature = F.dropout(in_feature, p = dropout_probability2)

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


def train(user_model, users, users_have_saw, user_history_dict, user_state_dict, user_history_train_dict, movies_tags, LOSS):
	# plt.ion()	# 实时画图
	# plt.show()
	loss_list = []

	optimizer = torch.optim.SGD(user_model.parameters(), lr=LR)
	for epoch_time in range(EPOCH):
		ii = 0
		epoch_plot = []
		# r_true_T_sum = 0
		# dis_exp_r_sum_T_sum = 0
		for uid, one_hot_arr in user_history_train_dict.items():
			for i in range(one_hot_arr.shape[1]//2):
				dis_exp_r_sum = 0
				r_true = 0

				true_click_one_hot = one_hot_arr[:, i].clone().view((-1, len(movies_tags)))
				state_one_hot_matrix = user_state_dict[uid]
				display_items_arr = sample(uid, movies_feature_dict, users_have_saw)	# list (len is K - 1)

				r_true = user_model(state_one_hot_matrix, true_click_one_hot)
				for item in display_items_arr:
					item = item.view((-1, len(movies_tags)))
					dis_exp_r_sum += torch.exp(eta*user_model(state_one_hot_matrix, item))
				dis_exp_r_sum += torch.exp(eta*r_true)
				# 将新的点击 item 加入state
				state_one_hot_matrix = torch.cat((state_one_hot_matrix[:, 1:], true_click_one_hot.t()), 1)

				# r_true_T_sum += r_true
				# dis_exp_r_sum_T_sum += dis_exp_r_sum_T_sum

				loss = LOSS(r_true, dis_exp_r_sum)
				optimizer.zero_grad()	# 先将梯度降为0
				loss.backward()			# 反向传递
				optimizer.step()		# 再用 optimizer 优化梯度	
				if ii%100 == 0:
					logging.debug('No. ' + str(ii) + ' Loss:' + str(loss))
				ii += 1
				loss_list.append(loss.data)

			# epoch_plot.append(sum([x.data for x in loss_list])/len(loss_list))

			# logging.debug('No. ' + str(ii) + ' Mean Loss:' + str(sum([x.data for x in loss_list])/len(loss_list)))
			print('No.', ii, "Mean Loss:", sum([x for x in loss_list])/len(loss_list))
			loss_list.clear()

		# print("mean Loss:", sum(epoch_plot)/len(epoch_plot))
		# plt.cla()
		# plt.plot(np.array([it for it in range(len(epoch_plot))]), epoch_plot, 'r-', lw=1)
		# plt.savefig("epoch"+str(epoch_time)+"loss.png")
		# plt.pause(0.1)
		# epoch_plot = []


def GAN_loss(r_true, dis_exp_r_sum):
	return (1/eta) * torch.log(dis_exp_r_sum + 1) - r_true


def test_user_module(user_model, users, users_have_saw, user_history_dict, user_state_dict, user_history_train_dict, movies_tags):
	right = 0
	sum_choose = 0
	for uid, one_hot_arr in user_history_train_dict.items():
		for i in range(one_hot_arr.shape[1]//2, one_hot_arr.shape[1]):
			true_click_one_hot = one_hot_arr[:, i].clone().view((-1, len(movies_tags)))
			state_one_hot_matrix = user_state_dict[uid]

			items_list = sample(uid, movies_feature_dict, users_have_saw)
			items_list.append(true_click_one_hot)
			random.shuffle(items_list)
			reward_list = np.array([user_model(state_one_hot_matrix, item.view((-1, len(movies_tags)))) for item in items_list])
			choose_item_index = reward_list.argmax()
			choose_item = items_list[choose_item_index]
			if torch.equal(choose_item, true_click_one_hot):
				right += 1
				# 将新的点击 item 加入state
				state_one_hot_matrix = torch.cat((state_one_hot_matrix[:, 1:], true_click_one_hot.t()), 1)
			sum_choose += 1
	return right/sum_choose


def greedy_recommend(user_model, users, users_have_saw, user_history_dict, user_state_dict, user_history_train_dict, movies_tags, movies_feature_dict):
	ct = 0
	rec_cnt = 0		# 推荐物品数
	for uid, _ in user_history_train_dict.items():
		rank_k_reward = []
		rank_k_mid = []
		rank_k_feature_vec = []
		state_one_hot_matrix = user_state_dict[uid]
		for _ in range(K):
			curr_rank_k_reward = [-99999]
			curr_rank_k_mid = [None]
			curr_rank_k_feature_vec = [None]
			for mid, feature in movies_feature_dict.items():
				feature = torch.tensor(feature, dtype=torch.float32).t()
				reward = user_model(state_one_hot_matrix, feature)
				if reward.data > curr_rank_k_reward[-1]:
					curr_rank_k_reward.pop()
					curr_rank_k_reward.append(reward)

					curr_rank_k_mid.pop()
					curr_rank_k_mid.append(mid)

					curr_rank_k_feature_vec.pop()
					curr_rank_k_feature_vec.append(feature)

			rank_k_reward.append(curr_rank_k_reward.pop())
			rank_k_mid.append(curr_rank_k_mid.pop())
			rank_k_feature_vec.append(curr_rank_k_feature_vec.pop())

		rec_cnt += K
		for mid, feature in zip(rank_k_mid, rank_k_feature_vec):
			if mid in users_have_saw[uid]:
				ct += 1
				# 将新的点击 item 加入state
				state_one_hot_matrix = torch.cat((state_one_hot_matrix[:, 1:], feature.view((len(movies_tags), -1))), 1)
		logging.debug("current CTR: " + str(ct/rec_cnt))
		print("current CTR: " + str(ct/rec_cnt), "CT:", ct, "uid:", uid)



start = time.time()

logging.basicConfig(level = logging.DEBUG,			# 控制台打印的日志级别
					filename = str(start)+'.log',
					filemode = 'a',					# 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
					# a是追加模式，默认如果不写的话，就是追加模式
					)

print('start! '+str(time.localtime(start)))
logging.debug('start! '+str(time.localtime(start)))
h_params = ['M:', M, 'N:', N, 'K:', K, 'eta:', eta, 'LR:', LR, 'EPOCH:', EPOCH, 'dropout1:', dropout_probability1, 'dropout2:', dropout_probability2, 'n_embedding:', n_embedding]
h_params = [str(x) for x in h_params]
logging.debug(' '.join(h_params))

users, movies, movies_tags, users_have_saw = process_data()
user_history_dict, user_state_dict, user_history_train_dict, movies_feature_dict = generate_data(users, movies, movies_tags)

user_model = UserModel_pw(len(movies_tags), M, N, 'elu', 256, 'elu', 128, 'elu', 1)

# item = user_history_train_dict[2][:, 1].clone().view((-1, len(movies_tags)))
# reward = user_model(user_state_dict[2], item)
# print(reward)

LOSS = GAN_loss
# LOSS = 
train(user_model, users, users_have_saw, user_history_dict, user_state_dict, user_history_train_dict, movies_tags, LOSS)

# 保存模型
torch.save(user_model, "user_model_by_Adam.pkl")
# 加载模型
# user_model = torch.load('user_model.pkl')

precision = test_user_module(user_model, users, users_have_saw, user_history_dict, user_state_dict, user_history_train_dict, movies_tags)
print('precision:', precision)
logging.debug('precision:' + str(precision))

greedy_recommend(user_model, users, users_have_saw, user_history_dict, user_state_dict, user_history_train_dict, movies_tags, movies_feature_dict)

logging.debug("END time:" + str(time.time() - start))
print(time.time() - start)