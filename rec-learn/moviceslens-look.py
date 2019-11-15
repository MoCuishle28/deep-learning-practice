import pandas as pd

"""
ratings 数据
文件里面的内容包含了每一个用户对于每一部电影的评分。数据格式如下：
	userId, movieId, rating, timestamp
	userId: 每个用户的id
	movieId: 每部电影的id
	rating: 用户评分，是5星制，按半颗星的规模递增(0.5 stars - 5 stars)
	timestamp: 自1970年1月1日零点后到用户提交评价的时间的秒数
	数据排序的顺序按照userId，movieId排列的。
"""

ratings = pd.read_csv("ml-latest-small/ratings.csv", index_col=None)
ratings.describe()    		# 什么用？好像去掉也行
# print(ratings.head(5))


"""
movies 数据
文件里包含了一部电影的id和标题，以及该电影的类别。数据格式如下：
	movieId, title, genres
	movieId:每部电影的id
	title:电影的标题
	genres:电影的类别（详细分类见readme.txt）
"""

movies = pd.read_csv("ml-latest-small/movies.csv", index_col=None)
# print(movies.head(5))

# 合并数据
data = pd.merge(ratings, movies, on='movieId')

# 汇总每部电影的评分数量，并降序排序
rating_count_by_movie = data.groupby(['movieId', 'title'], as_index=False)['rating'].count()
rating_count_by_movie.columns = ['movieId', 'title', 'rating_count']		# 不改名会默认第三列名字为 'rating'
rating_count_by_movie.sort_values(by=['rating_count'], ascending=False, inplace=True)
print(rating_count_by_movie[:10])

# 得到每部电影的均值和方差
rating_stddev = data.groupby(['movieId', 'title']).agg({'rating':['mean', 'std']})
print(rating_stddev.head(10))

genres = movies['genres']
genres_set = set()
for l in genres:
	ll = l.split('|')
	genres_set = genres_set | set(ll)

print(movies['genres'])
print(genres_set)
print(len(genres_set))	# 20