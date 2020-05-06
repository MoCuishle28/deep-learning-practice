import tensorflow as tf


class MLP(object):
	def __init__(self, args, name='MLP'):
		super(MLP, self).__init__()
		self.args = args
		# self.is_training = tf.placeholder(tf.bool, shape=())
		self.name = name
		with tf.variable_scope(self.name):
			self.inputs = tf.placeholder(tf.float32, [None, args.action_size], name='inputs')
			self.targets = tf.placeholder(tf.int32, [None], name='targets')

			self.logits = tf.contrib.layers.fully_connected(self.inputs, args.max_iid + 1, 
				weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))

			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, 
				logits=self.logits)
			self.loss = tf.reduce_mean(self.loss)
			self.optim = tf.train.AdamOptimizer(args.mlr).minimize(self.loss)