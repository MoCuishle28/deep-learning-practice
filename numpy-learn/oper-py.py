import numpy as np

a = np.array([[1, 2],
			[3, 4]])

b = np.array([[5, 6],
			[6, 5]])

print(a+b)
print(a-b)
print(a*b)	# 点乘
print(a/b)  # 点除
print('--------')
print(np.sqrt(a))

# 矩阵乘法
b = np.array([[1,2,3],
			[4,5,6]])
print(a.dot(b))
# 相同效果
print(np.dot(a, b))

b = np.array([2, 2])
b.reshape((2, 1))
print(a.dot(b))


# 常用函数
print('--------------')
print(np.sum(a))
# 对每一列求和
print(np.sum(a, axis=0))
# 对每一行求和
print(np.sum(a, axis=1))

# 求平均值 逻辑和sum一样
print(np.mean(a))
print(np.mean(a, axis=0))

# 生成随机数 3到4之间
print(np.random.uniform(3, 4))

# tile
print(a)
# 将矩阵 a 当作元素，一行两列
print(np.tile(a, (1,2)))
# 将矩阵 a 当作元素，两行三列
print(np.tile(a, (2,3)))

print('-------------')
# 排序
a = np.array([[3, 6, 4, 11],
			[5, 10, 1, 3]])
# 默认返回每一行排序的序号
print(a.argsort())
# 按照列排序，返回索引
print(a.argsort(axis=0))

print("-----------")
# 矩阵转置
print(a)
print(a.T)