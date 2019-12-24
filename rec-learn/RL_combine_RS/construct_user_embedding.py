import pickle
import argparse

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class UserEmbedding(nn.Module):
	def __init__(self, embedding_size, user_num):
		super(UserEmbedding, self).__init__()
		self.weight = torch.empty(embedding_size, user_num, dtype=torch.float32)
		self.weight.requires_grad_(requires_grad=True)
		nn.init.xavier_normal_(self.weight)


	def forward(self, x):
		# x: (batch, embedding_size)
		return torch.mm(x, self.weight)
		
		
def save_obj(obj, name):
	with open('../data/ml20/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open('../data/ml20/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


def set_uRow_mRow(users_behavior, movie_id_map_row):
	uid_map_uRow = {uid:row for row, uid in enumerate(users_behavior.keys())}
	mid_map_mRow = {mid:row for row, mid in enumerate(movie_id_map_row.keys())}
	# 0~609 610       0~4019 4020
	# print(min(uid_map_uRow.values()), max(uid_map_uRow.values()), len(uid_map_uRow.keys()))
	# print(min(mid_map_mRow.values()), max(mid_map_mRow.values()), len(mid_map_mRow.keys()))
	return uid_map_uRow, mid_map_mRow


def generate_rating_matrix(users_behavior, movie_id_map_row):
	# (610, 4020) -> 0~609, 0~4019
	rating_matrix = np.zeros((len(users_behavior.keys()), len(movie_id_map_row)), dtype=np.float32)
	# uid_map_uRow, mid_map_mRow = set_uRow_mRow(users_behavior, movie_id_map_row)
	del movie_id_map_row
	# save_obj(uid_map_uRow, 'uid_map_uRow')
	# save_obj(mid_map_mRow, 'mid_map_mRow')
	uid_map_uRow, mid_map_mRow = load_obj('uid_map_uRow'), load_obj('mid_map_mRow')

	for uid, behavior_list in users_behavior.items():
		for item in behavior_list:
			uRow = uid_map_uRow[uid]
			mRow = mid_map_mRow[item[0]]
			rating_matrix[uRow, mRow] = item[1]
	return rating_matrix


def normalize_rating_mattix(rating_matrix):
	norm_rating_matrix = np.zeros(rating_matrix.shape)
	# 需要正则化吗？ TODO
	return norm_rating_matrix


def generate_train_data(movie_embedding_matrix, batch, rating_matrix):
	for i in range(0, movie_embedding_matrix.shape[0], batch):
		if i+batch < movie_embedding_matrix.shape[0]:
			yield movie_embedding_matrix[i:i+batch, :], rating_matrix[:, i:i+batch].T
		else:
			yield movie_embedding_matrix[i:, :], rating_matrix[:, i:].T


def evaluate(user_embedding, rating_matrix, movie_embedding_matrix):
	movie_embedding_matrix = torch.tensor(movie_embedding_matrix, dtype=torch.float32)
	predict = user_embedding(movie_embedding_matrix)
	mask = rating_matrix != 0
	error = np.sum(np.abs((predict.t().detach().numpy() - rating_matrix)*mask))
	return error / (rating_matrix.shape[0]*rating_matrix.shape[1])


def train_UserEmbedding(args, user_embedding, rating_matrix, movie_embedding_128_mini):
	uid_map_uRow, mid_map_mRow = load_obj('uid_map_uRow'), load_obj('mid_map_mRow')
	uRow_map_uid = {uRow:uid for uid, uRow in uid_map_uRow.items()}
	mRow_map_mid = {mRow:mid for mid, mRow in mid_map_mRow.items()}

	optimizer = torch.optim.Adam([user_embedding.weight], lr=args.lr)
	lossFunc = nn.MSELoss()

	movie_embedding_matrix = np.zeros((len(mid_map_mRow.keys()), len(movie_embedding_128_mini[1])))
	for mid, embedding in movie_embedding_128_mini.items():
		row = mid_map_mRow[mid]
		movie_embedding_matrix[row, :] = embedding.detach().numpy()

	loss_list = []
	mean_error_list = []
	for i_epoch in range(args.epoch):
		for data, target in generate_train_data(movie_embedding_matrix, args.batch, rating_matrix):
			data = torch.tensor(data, dtype=torch.float32)
			target = torch.tensor(target, dtype=torch.float32)
			predict = user_embedding(data)

			mask = torch.tensor((target != 0).detach().numpy(), dtype=torch.float32)
			predict = predict * mask	# 只考虑已经评分的损失

			loss = lossFunc(predict, target)
			print('\r{}/{} LOSS:{:.6f}'.format(args.epoch, i_epoch+1, loss.item()), end='')
			loss_list.append(loss.item())

			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()

		print()
		mean_error = evaluate(user_embedding, rating_matrix, movie_embedding_matrix)
		print('mean error:{:.6f}'.format(mean_error))
		mean_error_list.append(mean_error)

	plt.subplot(1, 2, 1)
	plt.plot(range(len(loss_list)), loss_list)
	plt.subplot(1, 2, 2)
	plt.plot(range(len(mean_error_list)), mean_error_list)
	plt.show()


def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for FM")
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument('--epoch', type=int, default=20)
	parser.add_argument('--batch', type=int, default=128)
	args = parser.parse_args()

	movie_embedding_128_mini = load_obj('movie_embedding_128_mini')	# mid:embedding
	users_behavior = load_obj('users_rating')	# uid:[[mid, rating, timestamp], ...] value 内部要按照时间排序
	# movie_id_map_row = load_obj('movie_id_map_row')	# 这里的 row 对应了 csv 中的 row(需要构造对应矩阵的row)
	# users_rating = {k:{x[0]:x[1] for x in v} for k,v in users_behavior.items()}	# uid:{mid:rating, ...}

	# rating_matrix = generate_rating_matrix(users_behavior, movie_id_map_row)
	# np.save('../data/ml20/mini_rating_matrix.npy', rating_matrix)
	rating_matrix = np.load('../data/ml20/mini_rating_matrix.npy')

	user_embedding = UserEmbedding(len(movie_embedding_128_mini[1]), len(users_behavior.keys()))
	# train_UserEmbedding(args, user_embedding, rating_matrix, movie_embedding_128_mini)

	# user_embedding_matrix = user_embedding.weight.detach().numpy()
	# np.save('../data/ml20/user_embedding_matrix.npy', user_embedding_matrix)

	user_embedding_matrix = np.load('../data/ml20/user_embedding_matrix.npy')
	# 测试一下 embedding
	mid_map_mRow = load_obj('mid_map_mRow')
	movie_embedding_matrix = np.zeros((len(mid_map_mRow.keys()), len(movie_embedding_128_mini[1])))
	for mid, embedding in movie_embedding_128_mini.items():
		row = mid_map_mRow[mid]
		movie_embedding_matrix[row, :] = embedding.detach().numpy()
	movie_embedding_matrix = torch.tensor(movie_embedding_matrix, dtype=torch.float32)
	predict = torch.mm(movie_embedding_matrix, torch.tensor(user_embedding_matrix, dtype=torch.float32))
	mask = rating_matrix != 0
	error = np.sum(np.abs((predict.t().detach().numpy() - rating_matrix)*mask))
	print(error / (rating_matrix.shape[0]*rating_matrix.shape[1]))
	print(predict)
	print(rating_matrix)
	print(predict.shape)


if __name__ == '__main__':
	main()