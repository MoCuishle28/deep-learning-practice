import pickle

import torch
import numpy as np
import pandas as pd


def save_obj(obj, name):
	with open('../../data/ml20/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open('../../data/ml20/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


movie_id_map_row = {}		# mid:mRow
movie_id_map_row = load_obj('movie_id_map_row')
movie_genres = {}			# mid:[genres idx, genres idx, ...]
idx_map_genres = {}			# idx:genres
users_rating = {}			# uid:[[mid, rating, time], ...]


def get_movie_genres(movies_df, movie_id_map_row, idx_map_genres, movie_genres):
	genres_set = set()
	for mid in movie_id_map_row.keys():
		genres_list = movies_df[movies_df['movieId'] == mid]['genres'].values.tolist()[0].split('|')
		movie_genres[mid] = genres_list
		genres_set |= set(genres_list)

	genres_set = list(genres_set)
	idx_map_genres = {idx:genre for idx, genre in enumerate(genres_set)}
	genres_map_idx = {genre:idx for idx, genre in idx_map_genres.items()}

	for mid in movie_genres.keys():
		movie_genres[mid] = [genres_map_idx[genre] for genre in movie_genres[mid]]
	return movie_genres, idx_map_genres


# movies_df = pd.read_csv("../../ml-latest-small/ml-20m/movies.csv", index_col=None)
# print(movies_df)

# movie_genres, idx_map_genres = get_movie_genres(movies_df, movie_id_map_row, idx_map_genres, movie_genres)
# save_obj(movie_genres, 'movie_genres')
# save_obj(idx_map_genres, 'idx_map_genres')


######################################## 构造训练数据 ########################################
# mid: 1, 117590 之间不连续(归一化用到)
movie_genres, idx_map_genres = load_obj('movie_genres'), load_obj('idx_map_genres')
# print(idx_map_genres)		# idx: 0~18

users_rating = load_obj('users_rating')		# uid:[[mid, rating, time], ...]	uid: 1~610(归一化用到)

data = []		# uid, mid, gernes(19维 one hot) -> 21 维
target = []		# rating
for uid, behavior_list in users_rating.items():
	for behavior in behavior_list:
		x = np.zeros(21, dtype=np.int)
		x[0], x[1] = uid, behavior[0]		# uid, mid
		genes_list = movie_genres[behavior[0]]
		for idx in genes_list:
			x[idx+2] = 1
		data.append(x)
		target.append(behavior[1])				# rating

data = np.array(data)
target = np.array(target)

print(data)
print(target)
# np.save('../../data/ml20/mini_data.npy', data)
# np.save('../../data/ml20/mini_target.npy', target)

print('-------------------load-------------------')
data = np.load('../../data/ml20/mini_data.npy')
target = np.load('../../data/ml20/mini_target.npy')

print(data)
print(target)