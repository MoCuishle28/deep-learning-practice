import tensorflow as tf


# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())
print(random_float)

# 定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2))
print(zero_vector)

# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

# 查看矩阵A的形状、类型和值
print(A.shape)      # 输出(2, 2)，即矩阵的长和宽均为2
print(A.dtype)      # 输出<dtype: 'float32'>
print(A.numpy(), type(A.numpy()))

C = tf.add(A, B)    # 计算矩阵A和B的和
print(C)

D = tf.matmul(A, B) # 计算矩阵A和B的乘积
print(D)

print("----------自动求导机制----------")
x = tf.Variable(initial_value=3.)	# 初始化为3的变量
print(x.numpy())
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)				# y = x**2
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print([y, y_grad])


print('----------多多元函数、矩阵的求导---------')
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
	# tf.square() 操作代表对输入张量的每一个元素求平方，不改变张量形状
	# tf.reduce_sum() 操作代表对输入张量的所有元素求和，输出一个形状为空的纯量张量（可以通过 axis 参数来指定求和的维度，不指定则默认对所有元素求和）
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))		# (w*x+b - y)^2
w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数
print([L.numpy(), w_grad.numpy(), b_grad.numpy()])


print('-----------线性回归--------------')
import numpy as np
X_raw = np.array([i for i in range(2013, 2018)], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

# 归一化
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())


# 用 numpy 做线性回归
a, b = 0, 0

num_epoch = 10000
learning_rate = 1e-3
for e in range(num_epoch):
    # 手动计算损失函数关于自变量（模型参数）的梯度
    y_pred = a * X + b
    grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()

    # 更新参数
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

print(a, b)


# 用 tf 做线性回归
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数 (用 zip 函数将梯度和对应的变量拼在一起)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)