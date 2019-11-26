import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ratings = pd.read_csv('ratings.csv')
print(ratings)

# 存储成 pickle 格式文件
# ratings.to_pickle('ratings.pickle')

movies = pd.read_csv('movies.csv')
print(movies)

# 基于 index/columns 合并 (on='movieId' 基于 movieId 这个列合并，要按照多列合并则用 on = ['key1', 'key2',...])
# 默认的合并方法是 how = 'inner'		indicator=True 显示如何 merge 的（both 两个表都有的数据）
res = pd.merge(ratings, movies, on='movieId', indicator=True)
print(res)


print('-------------------')
# merge index, 按行 merge
df1 = pd.DataFrame({'A':[1,1,1], 'B':[2,2,2]}, index=['k0', 'k1', 'k2'])
df2 = pd.DataFrame({'C':[3,3,3], 'D':[4,4,4]}, index=['k0', 'k2', 'k3'])
print(df1)
print(df2)
# left_index, right_index 考虑 left、right 的 index
res = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
print(res)
res = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
print(res)

# data = pd.Series(np.random.randn(1000), index=np.arange(1000))
# data = data.cumsum()	# 生成的 1000 个随机数据的累加 (每一步都是前面的累加)
# data.plot()
# plt.show()

data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=['A', 'B', 'C', 'D'])
data = data.cumsum()
print(data.head())

# 会把 A,B,C,D 四组数据分别画上去
# data.plot()

# 画点的形式
ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label='Class 1')
# 将第二组赋在 ax 上
data.plot.scatter(x='A', y='C', color='red', label='Class 2', ax=ax)

plt.show()
# plt 的方法：bar, hist, box, kde, area, scatter, hexbin, pie, plot