import pandas as pd
import numpy as np


# 处理表格中的空值
dates = pd.date_range("20191123", periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan

print(df)

# drop 掉包含 nan 的行(axis=0)	{'any'->有 nan 就丢掉, 'all'->全是 nan 才丢掉}
print(df.dropna(axis=0, how='any'))
print(df.dropna(axis=0, how='all'))

# 填上 nan 为 0
print(df.fillna(value=0))

# 是否有缺失值
print(df.isnull())

# any 中至少包含了一个 true 则返回 true (至少有一个缺失值)
print(np.any(df.isnull()) == True)

df1 = pd.DataFrame(np.ones((6, 4))*2, index=dates, columns=['A', 'B', 'C', 'D'])
# 要改变 shape 哪一维的大小 axis 就等于几		(ignore_index=True 忽略原来的index，重新赋予列标号index)
print(pd.concat([df, df1], axis=0, ignore_index=True))

# join, ['inner', 'outer']
df1 = pd.DataFrame(np.ones((6, 4))*0, index=dates, columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.ones((6, 4))*2, index=dates, columns=['B', 'C', 'D', 'E'])

# 对于 axis=1 时则直接按列拼在一起
# columns 不一样，则合并时没有的 colums 会用 nan 填充
res = pd.concat([df1, df2], axis=0, join='outer')
print(res)
# inner 只合并有相同 columns 的部分，不同的部分裁剪掉
res = pd.concat([df1, df2], axis=0, join='inner')
print(res)

# 添加一行数据到表格
s = pd.Series([1,2,3,4], index=['A', 'B', 'C', 'D'])
df1 = df1.append(s, ignore_index=True)
print(df1)