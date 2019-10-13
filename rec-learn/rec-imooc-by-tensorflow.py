# 数据来源 https://grouplens.org/datasets/movielens/ 中的 ml-lastest-small.zip
# 要用 tensorflow 1.x
import pandas as pd
import numpy as np
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 用户对电影的评分
ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
# print(ratings_df.tail())


# 电影列表
movies_df = pd.read_csv("ml-latest-small/movies.csv")
# print(movies_df.tail())

# movieId 不等于行号
# 所以先添加行号到数据中作为 id
movies_df['movieRow'] = movies_df.index
# print(movies_df.tail())


# 筛选 movies_df 中的特征(只提取有用特征)
movies_df = movies_df[['movieRow', 'movieId', 'title']]
movies_df.to_csv('moviesProccessed.csv', index=False, header=True, encoding='utf-8')
# print(movies_df.tail())

# 要将 ratings_df 中的 movieId 替换成行号
# 合并电影列表矩阵和用户评分矩阵(每部电影名字及对应用户和用户评分)
ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
# print(ratings_df)

# 筛选特这个
ratings_df = ratings_df[['userId', 'movieRow', 'rating']]
ratings_df.to_csv('ratingsProcessed.csv', index=False, header=True, encoding='utf-8')
# print(ratings_df)
# row     userId  movieRow  rating
# 0            1         0     4.0
# 1            5         0     4.0


# 需要创建两个矩阵:电影评分矩阵 rating、用户是否已经评分 record
userNo = ratings_df['userId'].max()+1  		# 获取用户的最大编号
movieNo = ratings_df['movieRow'].max()+1	# 获取电影的最大编号

# 初始化评分矩阵 行号:电影最大数 列号:用户最大数    x(i,j) = 用户j对i的评分
rating = np.zeros((movieNo, userNo))
have_done = 0
ratings_df_len = np.shape(ratings_df)[0]

# 处理后保存到 txt 以后只需要 loadtxt
# row 获取的是每一行
# for index, row in ratings_df.iterrows():
# 	rating[int(row['movieRow']), int(row['userId'])] = row['rating']
# 	have_done += 1
# 	print('processed %d, %d left'%(have_done, ratings_df_len - have_done))
# np.savetxt('rating.txt', rating, fmt='%f',delimiter=',')

rating = np.loadtxt('rating.txt', delimiter=',')

print(rating)
# 大于0的地方表示已经有评分
record = rating > 0
# 将布尔值转为0/1 0表示未评分,1表示已评分
record = np.array(record, dtype=int)
print(record)


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
# 初始化电影内容矩阵 X; 用户矩阵 θ
# 随机初始化正太分布的 X
X_parameter = tf.Variable(tf.random_normal([movieNo, num_features], stddev=0.35))
Theta_parameter = tf.Variable(tf.random_normal([userNo, num_features], stddev=0.35))

# 定义代价函数. matmul是矩阵乘法, transpose_b=True->对θ进行转置, 乘以record是用0代替未评分电影
loss = 1/2 * tf.reduce_sum(((tf.matmul(X_parameter, Theta_parameter, transpose_b=True) - rating_norm) * record)**2) + 1/2 * (tf.reduce_sum(X_parameter**2) + tf.reduce_sum(Theta_parameter**2))

# 创建优化器 学习率 1e-4
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)


# 训练模型
# 可视化训练过程 (给 代价函数 起名 'loss')
tf.summary.scalar('loss', loss)
# 将 summary 信息汇总(这里只查看 loss 信息)
summaryMerged = tf.summary.merge_all()
filename = './movie_tensorboard'
writer = tf.summary.FileWriter(filename)
# 创建会话
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(5000):
	# train 的结果保存在 _, summaryMerged 训练结果保存到 movie_summary
	_, movie_summary = sess.run([train, summaryMerged])
    # tensorboard --logdir=./  查看 loss 变化
	writer.add_summary(movie_summary, i)

Current_X_parameters, Current_Theta_parameters = sess.run([X_parameter, Theta_parameter])
predicts = np.dot(Current_X_parameters, Current_Theta_parameters.T) + rating_mean
errors = np.sqrt(np.sum(predicts - rating)**2)
print(errors)


user_id = input("请输入用户编号:")
sortedResult = predicts[:, int(user_id)].argsort()[::-1]
idx = 0
print("推荐的20部电影是:".center(80, '='))
for i in sortedResult:
	print("评分:%.2f, 电影名:%s"%(predicts[i, int(user_id)], movies_df.iloc[i]['title']))
	idx+=1
	if idx == 20:
		break