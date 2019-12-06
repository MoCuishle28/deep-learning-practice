import pickle

import pandas as pd
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers


'''
		userId  movieId  rating   timestamp
0            1        1     4.0   964982703
1            1        3     4.0   964981247
2            1        6     4.0   964982224
3            1       47     5.0   964983815
4            1       50     5.0   964982931
...        ...      ...     ...         ...
100831     610   166534     4.0  1493848402
100832     610   168248     5.0  1493850091
100833     610   168250     5.0  1494273047
100834     610   168252     5.0  1493846352
100835     610   170875     3.0  1493846415

	  movieId                                      title                                       genres
0           1                           Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
1           2                             Jumanji (1995)                   Adventure|Children|Fantasy
2           3                    Grumpier Old Men (1995)                               Comedy|Romance
3           4                   Waiting to Exhale (1995)                         Comedy|Drama|Romance
4           5         Father of the Bride Part II (1995)                                       Comedy
...       ...                                        ...                                          ...
9737   193581  Black Butler: Book of the Atlantic (2017)              Action|Animation|Comedy|Fantasy
9738   193583               No Game No Life: Zero (2017)                     Animation|Comedy|Fantasy
9739   193585                               Flint (2017)                                        Drama
9740   193587        Bungo Stray Dogs: Dead Apple (2018)                             Action|Animation
9741   193609        Andrew Dice Clay: Dice Rules (1991)                                       Comedy
'''

# ratings_pd = pd.read_csv("ml-latest-small/ratings.csv", index_col=None)
# movies_pd = pd.read_csv("ml-latest-small/movies.csv", index_col=None)

tags_dict = {}			# tags_name: index
user_click = {}			# uid:[mid 1, mid 2, ...]   评分过的电影 (需要根据时间排序)
user_has_clicked = {}	# uid:{mid 1, mid 2, ...}   评分过的电影	(标记是否点击过)
movies_feature = {}		# mid:one-hot(np)	电影的 tags
no_geres_mid = set()	# 没有标签的 mid
movies_title_dict = {}	# mid:[title, genres]
tags_movies = {}		# tags_name:[mid 1, mid 2, ...]	同一个 tag 的电影


def get_movies_tags(tags_dict, no_geres_mid):
	movies_tags = set()
	for genres, mid in zip(movies_pd['genres'], movies_pd['movieId']):
		t = set(genres.split('|'))
		if '(no genres listed)' in t:
			no_geres_mid.add(mid)
		else:
			movies_tags = movies_tags | t

	for i, tag in enumerate(list(movies_tags)):
		tags_dict[tag] = i

	return tags_dict, no_geres_mid


def get_movies_feature(movies_feature, tags_dict, no_geres_mid, movies_title_dict, tags_movies):
	tags_movies = {tag_name:[] for tag_name, _ in tags_dict.items()}
	for mid, title, genres in zip(movies_pd['movieId'], movies_pd['title'], movies_pd['genres']):
		movies_title_dict[mid] = [title, genres]
		one_hot = np.zeros(len(tags_dict))
		if mid not in no_geres_mid:
			geres_list = genres.split('|')
			idxs = [tags_dict[tag_name] for tag_name in geres_list]
			one_hot[idxs] = 1
			movies_feature[mid] = one_hot
			for tag_name in geres_list:
				tags_movies[tag_name].append(mid)

	return movies_feature, movies_title_dict, tags_movies



def get_user_click(user_click, user_has_clicked, no_geres_mid):
	global ratings_pd
	ratings_pd = ratings_pd.sort_values(by='timestamp')
	for uid, mid in zip(ratings_pd['userId'], ratings_pd['movieId']):
		if uid not in user_click and mid not in no_geres_mid:
			user_click[uid] = []
			user_has_clicked[uid] = set()
		user_click[uid].append(mid)
		user_has_clicked[uid].add(mid)

	return user_click, user_has_clicked


def save_obj(obj, name):
	with open('data/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open('data/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


def save_data():
	save_obj(tags_dict, 'tags_dict')
	save_obj(no_geres_mid, 'no_geres_mid')
	save_obj(movies_feature, 'movies_feature')
	save_obj(movies_title_dict, 'movies_title_dict')
	save_obj(user_click, 'user_click')
	save_obj(user_has_clicked, 'user_has_clicked')
	save_obj(tags_movies, 'tags_movies')


# tags_dict, no_geres_mid = get_movies_tags(tags_dict, no_geres_mid)

# movies_feature, movies_title_dict, tags_movies = get_movies_feature(movies_feature, tags_dict, no_geres_mid, movies_title_dict, tags_movies)

# # 看过最多电影 2698, 最少 20
# user_click, user_has_clicked = get_user_click(user_click, user_has_clicked, no_geres_mid)

# save_data()


# 加载数据
# tags_dict = load_obj('tags_dict')
# no_geres_mid = load_obj('no_geres_mid')
# movies_feature = load_obj('movies_feature')
# movies_title_dict = load_obj('movies_title_dict')
# user_click = load_obj('user_click')
# user_has_clicked = load_obj('user_has_clicked')
# tags_movies = load_obj('tags_movies')


# 处理 embedding
# (moviesNo, 10)
# movies_embedding = np.load('models/X_parameter_withoutNorm.npy')

# col: 'movieRow', 'movieId', 'title'
# movies_df = pd.read_csv('data/moviesProccessed.csv')

movieId_to_MovieRow = {}	#  mid : movieRow
# save_obj(movieId_to_MovieRow, 'movieId_to_MovieRow')

# (9742, 10)
# print(movies_embedding.shape)


# user_click = load_obj('user_click')						# uid: [mid 1, mid 2, ...]
# movieId_to_MovieRow = load_obj('movieId_to_MovieRow')	# mid: movieRow

# min_ = 9999999
# max_ = 0
# for _, v in user_click.items():
# 	min_ = len(v) if len(v) < min_ else min_
# 	max_ = len(v) if len(v) > max_ else max_
# print(min_, max_)		# 20~2698

user_click_movieRow = {}		# uid: [movieRow 1, movieRow 2, ...]
# save_obj(user_click_movieRow, 'user_click_movieRow')
# user_click_movieRow = load_obj('user_click_movieRow')

user_has_clicked_movieRow = {}	# uid: {row 1, row 2, ...}
# save_obj(user_has_clicked_movieRow, 'user_has_clicked_movieRow')
# user_has_clicked_movieRow = load_obj('user_has_clicked_movieRow')


# 验证
# right = True
# for uid, row_set in user_has_clicked_movieRow.items():
# 	for row in user_click_movieRow[uid]:
# 		if row not in row_set:
# 			right = False
# 			break
# 	if not right:
# 		print(uid)
# 		break
# print('right' if right else 'not right')


# 验证
# row_to_idx = {v:k for k, v in movieId_to_MovieRow.items()}
# right = True
# for uid, row_list, mid_list in zip(user_click.keys(), user_click_movieRow.values(), user_click.values()):
# 	for row, mid in zip(row_list, mid_list):
# 		if row_to_idx[row] != mid:
# 			right = False
# 			break
# 	if not right:
# 		print(uid)
# 		break
# print('right' if right else 'not right')