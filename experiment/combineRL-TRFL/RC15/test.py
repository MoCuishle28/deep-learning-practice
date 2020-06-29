import tensorflow as tf


class A(object):
	def __init__(self):
		super(A, self).__init__()
		self.eval = tf.placeholder(tf.bool, shape=())
		self.a = tf.placeholder(tf.float32)
		self.b = tf.cond(tf.equal(self.eval, True), lambda: self.a + 99, lambda: self.a - 1)

# a =A()
# with tf.Session() as sess:
# 	t = sess.run(a.b, feed_dict={a.a: 1, a.eval: False})
# 	print(t)	# 0

# 	t = sess.run(a.b, feed_dict={a.a: 1, a.eval: True})
# 	print(t)	# 2

# 	t = sess.run(a.b, feed_dict={a.a: 1, a.eval: False})
# 	print(t)	# 0


class B(object):
	def __init__(self):
		super(B, self).__init__()
		self.x1 = tf.placeholder(tf.float32, [None, 3])
		self.x2 = tf.placeholder(tf.float32, [None, 3])
		self.inputs = tf.concat([self.x1, self.x2], axis=-1)
		self.atten = tf.contrib.layers.fully_connected(self.inputs, 2, activation_fn=None)
		# tf.nn.softmax()
		print(tf.expand_dims(self.atten[:, 0], -1))

		self.y = self.x1*tf.expand_dims(self.atten[:, 0], -1) + self.x2*tf.expand_dims(self.atten[:, 1], -1)
		print('inputs:', self.inputs)
		print('atten:', self.atten)
		print('y:', self.y)

class C(object):
	def __init__(self):
		self.x1 = tf.placeholder(tf.float32, [None, 3])
		self.x2 = tf.placeholder(tf.float32, [None, 3])
		self.inputs = tf.concat([self.x1, self.x2], axis=-1)
		self.atten = tf.contrib.layers.fully_connected(self.inputs, 6, activation_fn=None)
		self.atten = 6*tf.nn.softmax(self.atten)
		print(self.atten[:, :3])

		self.y = self.x1*self.atten[:, :3] + self.x2*self.atten[:, 3:]
		print('---')
		print(self.x1)
		print(self.atten[:, :3])
		print(self.x1*self.atten[:, :3])
		print('---')
		print('inputs:', self.inputs)
		print('atten:', self.atten)
		print('y:', self.y)


# b = B()
b = C()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	inputs, y, atten = sess.run([b.inputs, b.y, b.atten], 
		feed_dict={
		b.x1:[[1,2,3], [0,0.,9]], 
		b.x2:[[3,4.,5], [9,9.,5]]
		})

	print(inputs)
	print('---')
	print('atten:', atten)
	print('atten sum:', atten.sum(axis=1))
	print('---')
	print(y)

# # element-wise mult
# x = tf.constant([[1,2,3], [4,5,6.]])
# b = tf.constant([[9], [6.]])
# print(x)
# print(b)

# y = x*b
# with tf.Session() as sess:
# 	res = sess.run(y)
# 	print(res)