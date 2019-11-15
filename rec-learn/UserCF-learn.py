import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import math


movicesPath = "ml-latest-small/movies.csv"
ratingsPath = "ml-latest-small/ratings.csv"

moviesDF = pd.read_csv(movicesPath, index_col=None)
ratingsDF = pd.read_csv(ratingsPath, index_col=None)

# 按 4：1 划分训练集和测试集
trainRatingsDF, testRatingsDF = train_test_split(ratingsDF, test_size=0.2)
# print("total_movie_count:"+str(len(set(ratingsDF['movieId'].values.tolist()))))
# print("total_user_count:" + str(len(set(ratingsDF['userId'].values.tolist()))))
# print("train_movie_count:" + str(len(set(trainRatingsDF['movieId'].values.tolist()))))
# print("train_user_count:" + str(len(set(trainRatingsDF['userId'].values.tolist()))))
# print("test_movie_count:" + str(len(set(testRatingsDF['movieId'].values.tolist()))))
# print("test_user_count:" + str(len(set(testRatingsDF['userId'].values.tolist()))))

# 只要 ['userId','movieId','rating'] 这三列
"""
values 	是要进行汇总、统计等运算的列，可以是多个（list格式）
index  	是作为新表的列名，可以是多个（list格式）
columns	是作为新表的列名，可以是多个（list格式）
"""
trainRatingsPivotDF = pd.pivot_table(trainRatingsDF[['userId','movieId','rating']],columns=['movieId'],index=['userId'],values='rating',fill_value=0)
# print(trainRatingsPivotDF)

moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))

usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))

ratingValues = trainRatingsPivotDF.values.tolist()
# 每个元素是一个 list，包含了一个用户对所有电影的评分


# 用户相似度计算 这里使用余弦相似度
def calCosineSimilarity(list1,list2):
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1,val2) in zip(list1,list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (math.sqrt(denominator1 * denominator2))

# 向量化的计算
def calCosineSimilarity_byVector(v1, v2):
	return np.dot(v1.T, v2) / math.sqrt(np.dot(v1.T, v1) * np.dot(v2.T, v2))


# 计算用户的相似矩阵 用户相似矩阵是一个上三角矩阵，对角线为0（自己和自己的相似度没意义），下三角和上三角值相同
userSimMatrix = np.zeros((len(ratingValues),len(ratingValues)),dtype=np.float32)
for i in range(len(ratingValues)-1):
    for j in range(i+1,len(ratingValues)):
        userSimMatrix[i,j] = calCosineSimilarity_byVector(np.array(ratingValues[i]), np.array(ratingValues[j]))
        userSimMatrix[j,i] = userSimMatrix[i,j]


# 找到与用户最相近的 K 个用户所喜好的物品进行推荐
K = 10
userMostSimDict = dict()
for i in range(len(ratingValues)):
    userMostSimDict[i] = sorted(enumerate(list(userSimMatrix[i])),key = lambda x:x[1],reverse=True)[:K]


# 得到10个相近的用户后，计算目标用户对每个没看过的电影的兴趣分 p(u, i)
userRecommendValues = np.zeros((len(ratingValues),len(ratingValues[0])),dtype=np.float32)
for i in range(len(ratingValues)):
    for j in range(len(ratingValues[i])):
        if ratingValues[i][j] == 0:			# 
            val = 0
            for (user,sim) in userMostSimDict[i]:		# 目标用户 i 的 K 个相近用户
                val += (ratingValues[user][j] * sim)
            userRecommendValues[i,j] = val


# 为每个用户推荐10部电影
userRecommendDict = dict()
for i in range(len(ratingValues)):
    userRecommendDict[i] = sorted(enumerate(list(userRecommendValues[i])),key = lambda x:x[1],reverse=True)[:10]


# 将推荐的结果转换为推荐列表之后，我们将推荐结果转换为二元组
# 这里要注意的是，我们一直使用的是索引，我们需要将索引的用户id和电影id转换为真正的用户id和电影id
# 这里我们前面定义的两个map就派上用场了
userRecommendList = []
for key,value in userRecommendDict.items():
    user = usersMap[key]
    for (movieId,val) in value:
        userRecommendList.append([user,moviesMap[movieId]])


recommendDF = pd.DataFrame(userRecommendList,columns=['userId','movieId'])
recommendDF = pd.merge(recommendDF,moviesDF[['movieId','title']],on='movieId',how='inner')
print(recommendDF.tail(10))