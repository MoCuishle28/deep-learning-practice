import pandas as pd
import numpy as np


s = pd.Series([1, 3, 6, np.nan, 44, 1])
print(s)

# 生成 6 个数据, 是日期递增的序列(作为 row)
dates = pd.date_range('20191128', periods=6)
print(dates)


print('---')
# 6行4列 (index->行)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)

# 默认情况下的 row、 col
print('---')
df_default = pd.DataFrame(np.arange(24).reshape((6, 4)))
print(df_default)

# 还可以用字典创建 DataFrame (key->col)


print(s.dtypes)
print('--')
print(df.dtypes, df.index, df.columns)
print(df_default.values)

# 描述这个数据表
print(df.describe())

# 当作矩阵做转置
print(df.T)

# 排序 axis = 1 -> 对列标签排序
print('---sort_index---')
print(df.sort_index(axis=1, ascending=False))
# 对行标签排序
print(df_default.sort_index(axis=0, ascending=False))
# 对 value 排序 对 c 列排序	(默认 axis=0)
print(df.sort_values(by='c'))
# 为什么 axis = 1 才是行元素排序？？？ TODO
print(df.sort_values(by='20191128', axis=1))

print('---打印对应的列(3种方式)---')
print(df['a'])
print(df.a)
print(df.loc[:, ['a', 'c']])
print('---行( 0:3 行)---')
print(df_default[0:3])
# 0 列
print(df_default[0])

print('--')
# 行是左右都闭合区间
print(df.loc['20191128':'20191130', ['a','c']])
print(df_default.loc[0:3, :])

# ix 可以混合筛选（loc是纯标签筛选）
# print(df.ix[0:3, ['a', 'c']])

# iloc 是索引筛选(ix会有警告)
print(df.iloc[0:3, [0, 2]])

# 用条件筛选
print('---条件筛选---')
# a 列中大于0的选出 但其他列也会显示出来
print(df[df['a']>0])

print('设置值')
df.iloc[0,0] = 999
print(df)
print('a列大于0的赋值为0')
df[df['a'] < 0] = 0
print(df)

print('c列中，对应a列位置等于0的改为999')
df.c[df['a'] == 0] = 999
print(df)

print('加上空的行')
df['e'] = np.nan
print(df)