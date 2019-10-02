import numpy as np

a = np.array([1,2,3])
print(a, type(a))

# 明确确定向量是行向量还是列向量  (行, 列)
# 1 表示整个数组只有1行, -1是占位符
print(a.shape)
a = a.reshape((1, -1))
print(a.shape)

print("-------------------")
a = np.array([1,2,3,4,5,6])
print(a, a.shape)
a = a.reshape((2, -1))
print(a.shape)
print(a)

a = a.reshape((-1, 2))
print(a)

print('-------------------')
print(a[2,0])
a[2,0] = 100
print(a)

print('---------------------')
a = np.zeros((3,3))
print(a)

a = np.ones((2,3))
print(a)

a = np.full((2,3), 7)
print(a)

print('------------------')
# 创建单位矩阵
a = np.eye(3)
print(a)

print('------------------')
# 随机矩阵
a = np.random.random((3, 4))
print(a)