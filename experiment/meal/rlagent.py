import tensorflow as tf
import trfl
import numpy as np

from NextItNetModules import *
from SASRecModules import *
from utils import *


class QNetwork(object):
	def __init__(self, args, state_size, name='QNetwork'):
		super(QNetwork, self).__init__()
		self.args = args
		self.state_size = state_size
		self.name = name
		with tf.variable_scope(self.name):
			self.is_training = tf.placeholder(tf.bool, shape=())
			self.inputs = tf.placeholder(tf.float32, shape=(None, self.state_size))

			self.discount = tf.placeholder(tf.float32, [None] , name="discount")
			self.reward = tf.placeholder(tf.float32, [None], name='reward')
			self.target = tf.placeholder(tf.float32, [None],name='target')
			# TODO


	def get_qnetwork_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


class DDPG(object):
	def __init__(self, args, state_size, teacher_num, max_action=1.0, name='DDPG', dqda_clipping=None, clip_norm=False):
		super(DDPG, self).__init__()
		self.args = args
		self.state_size = state_size
		self.teacher_num = teacher_num
		self.name = name
		with tf.variable_scope(self.name):
			self.is_training = tf.placeholder(tf.bool, shape=())
			self.inputs = tf.placeholder(tf.float32, shape=(None, self.state_size))

			self.discount = tf.placeholder(tf.float32, [None] , name="discount")
			self.reward = tf.placeholder(tf.float32, [None], name='reward')
			self.target = tf.placeholder(tf.float32, [None],name='target')
			
			# DDPG
			actor = eval(args.actor_layers)
			actor.append(self.teacher_num)
			with tf.variable_scope("actor"):
				self.actor_output = mlp(self.inputs, self.is_training, hidden_sizes=actor, 
					dropout_rate=args.rl_dropout_rate, 
					l2=tf.contrib.layers.l2_regularizer(args.weight_decay))
			self.actor_out_ = self.actor_output * max_action
			self.weights = tf.nn.softmax(self.actor_out_)

			self.critic_input = tf.concat([self.actor_out_, self.inputs], axis=1)
			critic = eval(args.critic_layers)
			critic.append(1)
			with tf.variable_scope("critic"):
				self.critic_output = mlp(self.critic_input, self.is_training, hidden_sizes=critic, 
					output_activation=None, dropout_rate=args.rl_dropout_rate, 
					l2=tf.contrib.layers.l2_regularizer(args.weight_decay))

			self.dpg_return = trfl.dpg(self.critic_output, self.actor_out_, 
				dqda_clipping=dqda_clipping, clip_norm=clip_norm)

			self.actor_loss = tf.reduce_mean(self.dpg_return.loss)
			self.actor_optim = tf.train.AdamOptimizer(args.alr).minimize(self.actor_loss)

			self.td_return = trfl.td_learning(tf.squeeze(self.critic_output), self.reward, 
				self.discount, self.target)
			self.critic_loss = tf.reduce_mean(self.td_return.loss)
			self.critic_optim = tf.train.AdamOptimizer(args.clr).minimize(self.critic_loss)

	def get_qnetwork_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]