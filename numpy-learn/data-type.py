import numpy as np

a = np.array([1, 2])
print(a.dtype)

a = np.array([1, 6.2])
print(a.dtype)

# 会去掉小数部分
b = np.array([1.2, 2.6], dtype=np.int64)
print(b.dtype, b)

# 只取a的整数部分
b = np.array(a, dtype=np.int64)
print(b.dtype, b)