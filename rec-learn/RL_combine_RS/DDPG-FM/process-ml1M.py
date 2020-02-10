import pickle

import numpy as np
import pandas as pd


def save_obj(obj, name):
	with open('../../data/new_ml_1M/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open('../../data/new_ml_1M/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


users_rating = {}		# uid:[[mid, rating, timestamp], ...] 	value 内部要按照时间排序
users_has_clicked = {}	# uid:{mid, mid, ...}
mid_map_mfeature = {}	# mid: one-hot feature (21维 -> mid, genre, genre, ...)
genres_map_idx = {}		# genre: idx

train_data = None		# np matrix -> 每行是22维向量(包括: uid, mid, genre, genre, ...)
train_target = None		# np vector -> 每个元素是 data 中每行特征向量对应的 rating
valid_data = None
valid_target = None
test_data = None
test_target = None


ratings_df = pd.read_csv("../../ml-latest-small/ratings.csv")
# print(ratings_df)

movies_df = pd.read_csv("../../ml-latest-small/movies.csv")
# print(movies_df)


def sort_user_rating(users_rating):
	for uid in users_rating.keys():
		users_rating[uid].sort(key=lambda item: item[-1])
	return users_rating


def get_user_ratings(ratings_df):
	users_rating = {}		# uid:[[mid, rating, timestamp], ...]
	users_has_clicked = {}	# uid:{mid, mid, ...}

	for uid, movieId, rating, timestamp in zip(ratings_df['userId'], ratings_df['movieId'], ratings_df['rating'], ratings_df['timestamp']):
		if uid not in users_rating:
			users_rating[uid] = [[movieId, rating, timestamp]]
			users_has_clicked[uid] = set([movieId])
		else:
			users_rating[uid].append([movieId, rating, timestamp])
			users_has_clicked[uid].add(movieId)
	# 按照时间排序
	return sort_user_rating(users_rating), users_has_clicked


# 已经保存
# users_rating, users_has_clicked = get_user_ratings(ratings_df)
# print(len(users_has_clicked))	# 610 个用户
# 用户评分的电影数 20~2698
# print(min([len(l) for l in users_has_clicked.values()]), max([len(l) for l in users_has_clicked.values()]))


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
	for mid, genres_str in zip(movies_df['movieId'], movies_df['genres']):
		genres_list = genres_str.split('|')
		one_hot = np.zeros(len(genres_map_idx)+1, dtype=np.uint)
		one_hot[0] = mid
		for genre in genres_list:
			one_hot[genres_map_idx[genre]+1] = 1
		mid_map_mfeature[mid] = one_hot
	return mid_map_mfeature


# 已经保存
# genres_map_idx = get_genres_map_idx(movies_df)
# print(genres_map_idx)

# 已经保存
# mid_map_mfeature = get_mid_map_mfeature(movies_df, genres_map_idx)
# print(len(mid_map_mfeature))	# 9742


def save_dict(users_rating, users_has_clicked, mid_map_mfeature, genres_map_idx):
	save_obj(users_rating, 'users_rating')
	save_obj(users_has_clicked, 'users_has_clicked')
	save_obj(mid_map_mfeature, 'mid_map_mfeature')
	save_obj(genres_map_idx, 'genres_map_idx')

# 保存
# save_dict(users_rating, users_has_clicked, mid_map_mfeature, genres_map_idx)


users_rating = load_obj('users_rating')
mid_map_mfeature = load_obj('mid_map_mfeature')


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

# train_data, train_target, valid_data, valid_target, test_data, test_target = get_data_and_target(users_rating, mid_map_mfeature)


def save_matrix(train_data, train_target, valid_data, valid_target, test_data, test_target):
	np.save('../../data/new_ml_1M/train_data.npy', train_data)
	np.save('../../data/new_ml_1M/train_target.npy', train_target)
	np.save('../../data/new_ml_1M/valid_data.npy', valid_data)
	np.save('../../data/new_ml_1M/valid_target.npy', valid_target)
	np.save('../../data/new_ml_1M/test_data.npy', test_data)
	np.save('../../data/new_ml_1M/test_target.npy', test_target)

# save_matrix(train_data, train_target, valid_data, valid_target, test_data, test_target)

# 每一行: uid, mid, genres 1, genres 2, ...
train_data = np.load('../../data/new_ml_1M/train_data.npy')
train_target = np.load('../../data/new_ml_1M/train_target.npy')
valid_data = np.load('../../data/new_ml_1M/valid_data.npy')
valid_target = np.load('../../data/new_ml_1M/valid_target.npy')
test_data = np.load('../../data/new_ml_1M/test_data.npy')
test_target = np.load('../../data/new_ml_1M/test_target.npy')

print(train_data.shape, train_target.shape)
print(valid_data.shape, valid_target.shape)
print(test_data.shape, test_target.shape)

print(train_data)
print(train_target)
print('---')
print(valid_data)
print(valid_target)
print('---')
print(test_data)
print(test_target)