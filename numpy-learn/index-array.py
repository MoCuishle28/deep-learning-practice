import numpy as np

a = np.array([[1,2,3,4],
			[5,6,7,8],
			[9,10,11,12]])

# 错误写法 a[-2:][1:3]
print(a[-2:, 1:3])
# [ [6, 7]
# 	[10, 11] ]

b = a[-2:, 1:3]
print(b.shape)
b = a[2, 1:3]
print(b.shape, b)
b = a[1, 2]
print(b.shape, b)

# 将前3行的1列元素加10
a[np.arange(3), 1] += 10
print(a)

print(np.arange(3), np.arange(3,7))

# 以上写法的好处:
# 获取矩阵中大于10的元素的结果
result = a > 10
print(result)
# 取出矩阵中大于10的元素
print(a[result])

# 以上操作简化
print(a[a > 10])