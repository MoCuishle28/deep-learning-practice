import pandas as pd
import numpy as np


# 数量: 1~610
# ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
# movies_df = pd.read_csv("ml-latest-small/movies.csv")

# movieId 不等于行号, 所以先添加行号到数据中作为 id (0~9741)
# movies_df['movieRow'] = movies_df.index

movies_df = pd.read_csv('data/moviesProccessed.csv')
ratings_df = pd.read_csv('data/ratingsProcessed.csv')
# # row     userId  movieRow  rating
# # 0            1         0     4.0
# # 1            5         0     4.0

userNo = ratings_df['userId'].max()  		# 获取用户的最大编号	个数：610(userId 从1开始, 不能+1)
movieNo = ratings_df['movieRow'].max()+1	# 获取电影的最大编号	个数：9742(row 从0开始, 要+1)

# rating = np.zeros((movieNo, userNo))

# row 获取的是每一行
# for index, row in ratings_df.iterrows():
# 	print(row)
# 	assert 0 > 1
# 	rating[int(row['movieRow']), int(row['userId'])] = row['rating']
# np.save('data/new_rating.npy', rating)

################### read rating ################
rating = np.load('data/rating.npy')
print(rating.shape)
print(rating[-1, :])	# 全0
print(rating[610, :])

# user_embedding = np.load('models/Theta_parameter_withoutNorm.npy')
# movie_embedding = np.load('models/X_parameter_withoutNorm.npy')

# print(user_embedding.shape)
# print(user_embedding[-1, :])
# print(movie_embedding.shape)
# print(movie_embedding[-1, :])