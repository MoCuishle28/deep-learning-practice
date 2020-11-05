import numpy as np
import tensorflow as tf
import trfl
import random


def mlp(x, is_training, hidden_sizes=(32,), activation=tf.nn.relu, output_activation=None, 
	dropout_rate=0.5, l2=None):
	for h in hidden_sizes[:-1]:
		x = tf.layers.dense(x, units=h, activation=activation, activity_regularizer=l2)
		x = tf.layers.dropout(x, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
	return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, activity_regularizer=l2)


from collections import namedtuple
Transition = namedtuple(
	'Transition', ('state', 'action', 'next_state', 'reward', 'is_done'))

class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class QNetwork(object):
	def __init__(self, args, state_size, item_num, name='QNetwork'):
		super(QNetwork, self).__init__()
		self.args = args
		self.state_size = state_size
		self.name = name
		self.item_num = item_num
		self.memory = ReplayMemory(args.maxlen)
		with tf.variable_scope(self.name):
			self.is_training = tf.placeholder(tf.bool, shape=())
			self.inputs = tf.placeholder(tf.float32, shape=(None, self.state_size))

			# Q-learning
			self.actions = tf.placeholder(tf.int32, [None])
			self.targetQs_ = tf.placeholder(tf.float32, [None, item_num])
			self.targetQs_selector = tf.placeholder(tf.float32, [None,
																 item_num])  # used for select best action for double q learning
			self.rewards = tf.placeholder(tf.float32, [None])
			self.discount = tf.placeholder(tf.float32, [None])

			sampler_layers = eval(args.sampler_layers)
			sampler_layers.append(item_num)
			l2 = None if args.weight_decay == 0 else tf.contrib.layers.l2_regularizer(args.weight_decay)
			with tf.variable_scope("qsampler"):
				self.output = mlp(self.inputs, self.is_training, 
					hidden_sizes=sampler_layers, output_activation=tf.nn.tanh,	# note
					dropout_rate=args.sampler_dropout_rate, l2=l2)
				self.output = self.output * self.args.max_q
			
			# TRFL double qlearning
			qloss, q_learning = trfl.double_qlearning(self.output, self.actions, self.rewards, self.discount,
													  self.targetQs_, self.targetQs_selector)
			self.loss = tf.reduce_mean(qloss)
			self.opt = tf.train.AdamOptimizer(self.args.qlr).minimize(self.loss)


	def get_qnetwork_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]