import pickle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data


def save_obj(obj, name):
	with open('data/ml_1M_row/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open('data/ml_1M_row/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


def sort_user_rating(users_rating):
	for uid in users_rating.keys():
		users_rating[uid].sort(key=lambda item: item[-1])
	return users_rating


def get_user_ratings(ratings_df):
	users_rating = {}		# uid:[[mrow, rating, timestamp], ...]
	users_has_clicked = {}	# uid:{mrow, mrow, ...}

	for uid, movieRow, rating, timestamp in zip(ratings_df['userId'], ratings_df['movieRow'], ratings_df['rating'], ratings_df['timestamp']):
		if uid not in users_rating:
			users_rating[uid] = [[movieRow, rating, timestamp]]
			users_has_clicked[uid] = set([movieRow])
		else:
			users_rating[uid].append([movieRow, rating, timestamp])
			users_has_clicked[uid].add(movieRow)
	# 按照时间排序
	return sort_user_rating(users_rating), users_has_clicked


def get_genres_map_idx(movies_df):
	genres_set = set()
	no_genres_num = 0
	for genres_str in movies_df['genres']:
		if genres_str == '(no genres listed)':
			no_genres_num += 1
		genres_list = genres_str.split('|')
		genres_set = genres_set | set(genres_list)
	# print(no_genres_num)	# 34部电影没有 genres
	genres_map_idx = {genre:i for i, genre in enumerate(genres_set)}	# genre: idx
	return genres_map_idx


def get_mid_map_mfeature(movies_df, genres_map_idx):
	mid_map_mfeature = {}	# mid: one-hot feature (21维 -> mid, genre, genre, ...)
	for mid, genres_str in zip(movies_df['movieRow'], movies_df['genres']):
		genres_list = genres_str.split('|')
		one_hot = np.zeros(len(genres_map_idx)+1, dtype=np.uint)
		one_hot[0] = mid
		for genre in genres_list:
			one_hot[genres_map_idx[genre]+1] = 1
		mid_map_mfeature[mid] = one_hot
	return mid_map_mfeature


def save_dict(users_rating, users_has_clicked, mid_map_mfeature, genres_map_idx):
	save_obj(users_rating, 'users_rating')
	save_obj(users_has_clicked, 'users_has_clicked')
	save_obj(mid_map_mfeature, 'mid_map_mfeature')
	save_obj(genres_map_idx, 'genres_map_idx')


def build_user_item_matrix(users_rating, mid_map_mfeature):
	max_row = max(users_rating.keys()) + 1
	max_col = max(mid_map_mfeature.keys()) + 1
	# row: uid(0~610), col: mrow(0~9741), element: rating
	user_item_matrix = np.zeros((max_row, max_col), dtype=np.float32)
	for uid, behavior in users_rating.items():
		for item in behavior:
			user_item_matrix[uid, item[0]] = item[1]
	return user_item_matrix


# 每个用户按照时间顺序 8：1：1 划分训练集、验证集、测试集
def get_data_and_target(users_rating, mid_map_mfeature, valid_rate=1, test_rate=1):
	train_data_list = []
	train_target_list = []
	valid_data_list = []
	valid_target_list = []
	test_data_list = []
	test_target_list = []
	for uid, behavior_list in users_rating.items():
		size = len(behavior_list)
		unit = size // 10
		# generate test data
		for i in range(test_rate * unit):
			item = behavior_list.pop()
			vector = np.concatenate([np.array([uid], dtype=np.uint), mid_map_mfeature[item[0]]])
			test_data_list.append(vector)
			test_target_list.append(item[1])
		# generate valid data
		for i in range(valid_rate * unit):
			item = behavior_list.pop()
			vector = np.concatenate([np.array([uid], dtype=np.uint), mid_map_mfeature[item[0]]])
			valid_data_list.append(vector)
			valid_target_list.append(item[1])
		# generate train data
		train_data_list.extend([np.concatenate([np.array([uid], dtype=np.uint), mid_map_mfeature[item[0]]]) for item in behavior_list])
		train_target_list.extend(item[1] for item in behavior_list)

	train_data = np.array(train_data_list)
	train_target = np.array(train_target_list)

	valid_data = np.array(valid_data_list)
	valid_target = np.array(valid_target_list)

	test_data = np.array(test_data_list)
	test_target = np.array(test_target_list)

	return train_data, train_target, valid_data, valid_target, test_data, test_target


def random_build_dataset():
	base_data_dir = 'data/ml_1M_row/'
	train_data = torch.tensor(np.load(base_data_dir + 'train_data.npy').astype(np.float32), dtype=torch.float32)
	train_target = torch.tensor(np.load(base_data_dir + 'train_target.npy').astype(np.float32), dtype=torch.float32)
	valid_data = torch.tensor(np.load(base_data_dir + 'valid_data.npy').astype(np.float32), dtype=torch.float32)
	valid_target = torch.tensor(np.load(base_data_dir + 'valid_target.npy').astype(np.float32), dtype=torch.float32)
	test_data = torch.tensor(np.load(base_data_dir + 'test_data.npy').astype(np.float32), dtype=torch.float32)
	test_target = torch.tensor(np.load(base_data_dir + 'test_target.npy').astype(np.float32), dtype=torch.float32)

	data = torch.cat([train_data, valid_data, test_data])
	target = torch.cat([train_target, valid_target, test_target])
	all_data = Data.TensorDataset(data, target)

	train_size, valid_size = train_data.shape[0], valid_data.shape[0]
	train_data, valid_data, test_data = torch.utils.data.random_split(all_data, [train_size, valid_size, valid_size])

	batch_size = 512
	train_data = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
	valid_data = Data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
	test_data = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

	d = {'train':train_data, 'valid':valid_data, 'test':test_data}
	for k, v in d.items():
		data_list = []
		target_list = []
		for data, target in v:
			data_list.append(data.numpy())
			target_list.append(target.numpy())
		data = np.concatenate(data_list)
		target = np.concatenate(target_list)
		np.save('data/ml_1M_row/without_time_seq/' + k + '_data.npy', data)
		np.save('data/ml_1M_row/without_time_seq/' + k + '_target.npy', target)
		# (81200, 22) (81200,)
		# (9818, 22) (9818,)
		# (9818, 22) (9818,)
		print(data.shape, target.shape)


def save_np(train_data, train_target, valid_data, valid_target, test_data, test_target, user_item_matrix):
	np.save('data/ml_1M_row/train_data.npy', train_data)
	np.save('data/ml_1M_row/train_target.npy', train_target)
	np.save('data/ml_1M_row/valid_data.npy', valid_data)
	np.save('data/ml_1M_row/valid_target.npy', valid_target)
	np.save('data/ml_1M_row/test_data.npy', test_data)
	np.save('data/ml_1M_row/test_target.npy', test_target)
	np.save('data/ml_1M_row/user_item_matrix', user_item_matrix)


def main():
	# uid = 1~610
	users_rating = {}		# uid:[[mrow, rating, timestamp], ...] 	value 内部要按照时间排序
	users_rating_without_timestamp = {}	# uid:[[mrow, rating], ...] 	value 内部要按照时间排序
	users_has_clicked = {}	# uid:{mrow, mrow, ...}
	# mrow = 0~9741
	mid_map_mfeature = {}	# mrow: one-hot feature (21维 -> mrow, genre, genre, ...)
	genres_map_idx = {}		# genre: idx
	# shape -> (611, 9742)
	user_item_matrix = None 	# row: uid(0~610), col: mrow(0~9741), element: rating

	train_data = None		# np matrix -> 每行是22维向量(包括: uid, mrow, genre, genre, ...)
	train_target = None		# np vector -> 每个元素是 data 中每行特征向量对应的 rating
	valid_data = None
	valid_target = None
	test_data = None
	test_target = None

	# ratings_df = pd.read_csv("data/ml-latest-small/ratings.csv")
	# movies_df = pd.read_csv("data/ml-latest-small/movies.csv")
	# movies_df['movieRow'] = movies_df.index
	# movies_df = movies_df[['movieRow', 'movieId', 'genres']]

	# ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
	# ratings_df = ratings_df[['userId', 'movieRow', 'rating', 'timestamp']]

	# # save
	# users_rating, users_has_clicked = get_user_ratings(ratings_df)
	# # save
	# genres_map_idx = get_genres_map_idx(movies_df)
	# # save
	# mid_map_mfeature = get_mid_map_mfeature(movies_df, genres_map_idx)
	# save_dict(users_rating, users_has_clicked, mid_map_mfeature, genres_map_idx)
	# ------------------------------------以上为 dict 数据的处理------------------------------------

	# users_rating = load_obj('users_rating')
	# mid_map_mfeature = load_obj('mid_map_mfeature')

	# user_item_matrix = build_user_item_matrix(users_rating, mid_map_mfeature)

	# train_data, train_target, valid_data, valid_target, test_data, test_target = get_data_and_target(users_rating, mid_map_mfeature)
	# save_np(train_data, train_target, valid_data, valid_target, test_data, test_target, user_item_matrix)
	# -------------------------------------以上为 np 数据的保存-------------------------------------
	
	# # (81200, 22) (81200,)
	# train_data = np.load('data/ml_1M_row/train_data.npy')
	# train_target = np.load('data/ml_1M_row/train_target.npy')
	# # (9818, 22) (9818,)
	# valid_data = np.load('data/ml_1M_row/valid_data.npy')
	# valid_target = np.load('data/ml_1M_row/valid_target.npy')
	# test_data = np.load('data/ml_1M_row/test_data.npy')
	# test_target = np.load('data/ml_1M_row/test_target.npy')

	# 构建无时间序列的数据
	# random_build_dataset()

	# users_rating = load_obj('users_rating')
	# for uid, behavior_list in users_rating.items():
	# 	for item in behavior_list:
	# 		item.pop()
	# save_obj(users_rating, 'users_rating_without_timestamp')

if __name__ == '__main__':
	main()