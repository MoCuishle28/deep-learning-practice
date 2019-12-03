import os
import logging
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level = logging.DEBUG, filename = 'log/CF-log.log', filemode = 'a')
log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logging.debug('start time: '+ str(log_time))

# ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
# movies_df = pd.read_csv("ml-latest-small/movies.csv")

# # movieId 不等于行号
# # 所以先添加行号到数据中作为 id
# movies_df['movieRow'] = movies_df.index


# # 筛选 movies_df 中的特征(只提取有用特征)
# movies_df = movies_df[['movieRow', 'movieId', 'title']]
# movies_df.to_csv('data/moviesProccessed.csv', index=False, header=True, encoding='utf-8')

# # 要将 ratings_df 中的 movieId 替换成行号
# # 合并电影列表矩阵和用户评分矩阵 (每部电影名字及对应用户和用户评分)
# ratings_df = pd.merge(ratings_df, movies_df, on='movieId')

# # 筛选特这个
# ratings_df = ratings_df[['userId', 'movieRow', 'rating']]
# ratings_df.to_csv('data/ratingsProcessed.csv', index=False, header=True, encoding='utf-8')
# row     userId  movieRow  rating
# 0            1         0     4.0
# 1            5         0     4.0


# 保存好了直接读取
# movies_df = pd.read_csv('data/moviesProccessed.csv')

# ratings_df = pd.read_csv('data/ratingsProcessed.csv')

# # # 需要创建两个矩阵:电影评分矩阵 rating、用户是否已经评分 record
# userNo = ratings_df['userId'].max()+1  		# 获取用户的最大编号
# movieNo = ratings_df['movieRow'].max()+1	# 获取电影的最大编号
# print(userNo, movieNo)
# assert 0 > 1	看完 userNo, movieNo 直接中断

# # 初始化评分矩阵 行号:电影最大数 列号:用户最大数    x(i,j) = 用户j对i的评分
# rating = np.zeros((movieNo, userNo))

# # 处理后保存到 txt 以后只需要 loadtxt
# # row 获取的是每一行
# for index, row in ratings_df.iterrows():
# 	rating[int(row['movieRow']), int(row['userId'])] = row['rating']
# np.save('data/rating.npy', rating)

userNo = 611
movieNo = 9742

rating = np.load('data/rating.npy')
record = rating > 0						# 大于0的地方表示已经有评分
record = np.array(record, dtype=int)	# 将布尔值转为 0/1  0表示未评分,1表示已评分


# 将评分取值范围缩放(归一化?)
def normalizeRating(rating, record):
	# 电影数量, 用户数量
	m, n = rating.shape
	rating_mean = np.zeros((m, 1))
	# 处理后的 rating 矩阵
	rating_norm = np.zeros((m, n))
	# 原始评分 - 平均评分
	for i in range(m):
		# 获取已经评分电影 i 的用户的下标
		idx = record[i, :] != 0
		# 电影 i 的评价分
		rating_mean[i] = np.mean(rating[i, idx])
		rating_norm[i, idx] -= rating_mean[i]
	return rating_norm, rating_mean


rating_norm, rating_mean = normalizeRating(rating, record)
# 全部是0的行会变为NAN, 需要处理
rating_norm = np.nan_to_num(rating_norm)
rating_mean = np.nan_to_num(rating_mean)


# 假设有10种类型的电影
num_features = 10
learning_rate = 1e-4
EPOCH = 100

class CF(tf.keras.Model):
	def __init__(self, movieNo, userNo, num_features):
		super().__init__()
		# 初始化电影内容矩阵 X; 用户矩阵 θ
		# 随机初始化正太分布的 X
		self.X_parameter = tf.Variable(tf.random.normal([movieNo, num_features], stddev=0.35))
		self.Theta_parameter = tf.Variable(tf.random.normal([userNo, num_features], stddev=0.35))


	def call(self, input):
		return tf.matmul(self.X_parameter, self.Theta_parameter, transpose_b=True)
		

model = CF(movieNo, userNo, num_features)

# 加载模型
# model.load_weights('models/cf_model_weight')

# 定义代价函数. matmul是矩阵乘法, transpose_b=True->对θ进行转置, 乘以record是用 0 代替未评分电影
def Loss(predict_rating):
	return 1/2 * tf.reduce_sum(((predict_rating - rating_norm) * record)**2) + 1/2 * (tf.reduce_sum(model.X_parameter**2) + tf.reduce_sum(model.Theta_parameter**2))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)


# 训练模型
for _ in range(EPOCH):
	with tf.GradientTape() as tape:
		y_pred = model([])
		loss = Loss(y_pred)
	grads = tape.gradient(loss, model.variables)
	optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

# 保存模型参数
model.save_weights('models/cf_model_weight')

Current_X_parameters, Current_Theta_parameters = model.X_parameter.numpy(), model.Theta_parameter.numpy()
del model


predicts = np.dot(Current_X_parameters, Current_Theta_parameters.T) + rating_mean
errors = np.sqrt(np.sum(predicts - rating)**2)

# EPOCH = 100 -> Error = 19030317.181246735
print('ERROR:')
print(errors)

logging.debug('ERROR: '+ str(errors))
print(Current_X_parameters)