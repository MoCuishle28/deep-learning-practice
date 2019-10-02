import numpy as np

a = np.array([[1,2,3],
			[2,3,4],
			[12,31,22],
			[2,2,2]])

b = np.array([1,2,3])

# 想实现每一行加上b
for i in range(4):
	a[i,:] += b
print(a)
print('-------------')

a = np.array([[1,2,3],
			[2,3,4],
			[12,31,22],
			[2,2,2]])
print(a + np.tile(b, (4,1)))
print('-------------')

# 广播实现比较高效
# 可以直接相加
print(a + b)