import pickle

import torch
import numpy as np
import pandas as pd


# genome_scores = pd.read_csv("../../ml-latest-small/ml-20m/genome-scores.csv", index_col=None)
# print(genome_scores)
#  movieId  tagId  relevance
#  1      	1    	0.02500
#  1      	2    	0.02500
#  1      	3    	0.05775

# genome_tags = pd.read_csv("../../ml-latest-small/ml-20m/genome-tags.csv", index_col=None)
# print(genome_tags)
# tagId         tag
# 1           	007
# 2  			007 (series)


# 是 20M 的数据集的 embedding  (key 是movieId; value 的类型是 torch.Tensor)
embedding_20M_path = 'ml_embedding_pca128'
users_rating = {}		# uid:[[mid, rating, timestamp], ...] 	value 内部要按照时间排序
movie_id_map_row = {}	# mid:mRow
movie_embedding_128_mini = {}	# mid:embedding (embedding 是 pytorch.Tensor)


def save_obj(obj, name):
	with open('../../data/ml20/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open('../../data/ml20/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

def get_users(ratings_df, user_num=610):
	# 只取 610 个用户 (根据 1M 的 ml 来选数据规模)
	user_click = load_obj('user_click')
	click_size_list = []	# 
	for mid_list in user_click.values():
		click_size_list.append(len(mid_list))
	del user_click

	users_rating = {}		# uid:[[mid, rating, timestamp], ...]
	contain_movie = set()	# 包含电影的 ID
	curr_user_num = 0
	for uid, movieId, rating, timestamp in zip(ratings_df['userId'], ratings_df['movieId'], ratings_df['rating'], ratings_df['timestamp']):
		if curr_user_num == user_num:
			break

		if uid not in users_rating:
			users_rating[uid] = [[movieId, rating, timestamp]]
			contain_movie.add(movieId)
			curr_user_num += 1
		elif len(users_rating[uid]) < click_size_list[curr_user_num]:
			users_rating[uid].append([movieId, rating, timestamp])
			contain_movie.add(movieId)

	return users_rating, contain_movie


def sort_user_rating(users_rating):
	for uid in users_rating.keys():
		users_rating[uid].sort(key=lambda item: item[-1])
	return users_rating


def get_mid_map_mRow(movies_df, contain_movie):
	movie_id_map_row = {}	# mid:mRow
	for mid in contain_movie:
		row = movies_df.movieRow[movies_df['movieId'] == mid].index.tolist()[-1]
		movie_id_map_row[mid] = row
	return movie_id_map_row


############################ 初步裁剪 20M 数据 ############################
ratings_df = pd.read_csv("../../ml-latest-small/ml-20m/ratings.csv", index_col=None)
# print(ratings_df)

users_rating, contain_movie = get_users(ratings_df)			# uid: 1~610, sum movie num:37721
del ratings_df
users_rating = sort_user_rating(users_rating)	# 按照时间排序

movies_df = pd.read_csv("../../ml-latest-small/ml-20m/movies.csv")
# movieId 不等于行号, 所以先添加行号到数据中作为 id
movies_df['movieRow'] = movies_df.index
# print(movies_df)

movie_id_map_row = get_mid_map_mRow(movies_df, contain_movie)

save_obj(movie_id_map_row, 'movie_id_map_row')
save_obj(users_rating, 'users_rating')

# 设计用户 610(uid:1~610), 涉及 4020 部电影, sum movie num:37721(每个用户看的电影和)
print('user num:{}, movie num:{}'.format(len(users_rating.keys()), len(movie_id_map_row)))


############################ 提取 embedding ############################
all_embedding_data = load_obj(embedding_20M_path)	# mid: embedding
for mid in movie_id_map_row.keys():
	movie_embedding_128_mini[mid] = all_embedding_data[mid]

save_obj(movie_embedding_128_mini, 'movie_embedding_128_mini')
print('movie embedding num:',len(movie_embedding_128_mini.keys()))


############################ 划分 train, valid, test ############################
# 8:1:1
# TODO