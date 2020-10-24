import tensorflow as tf
import numpy as np

from NextItNetModules import *
from SASRecModules import *
from utils import *


class GRUnetwork:
	def __init__(self, args, hidden_size, learning_rate, dlr, item_num, state_size, embeddings, name='gru_teacher'):
		self.args = args
		self.state_size = state_size
		self.learning_rate = learning_rate
		self.dlr = dlr
		self.hidden_size=hidden_size
		self.item_num=int(item_num)
		self.hw = state_size
		self.name = name
		with tf.variable_scope(self.name):
			self.is_training = tf.placeholder(tf.bool, shape=())

			# all_embeddings=self.initialize_embeddings()
			all_embeddings = embeddings

			self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
			self.len_state=tf.placeholder(tf.int32, [None],name='len_state')
			self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss

			self.input_emb=tf.nn.embedding_lookup(all_embeddings['state_embeddings'],self.inputs)

			gru_out, self.states_hidden= tf.nn.dynamic_rnn(
				tf.contrib.rnn.GRUCell(self.hidden_size),
				self.input_emb,
				dtype=tf.float32,
				sequence_length=self.len_state,
			)
			if args.layer_trick == 'ln':
				self.states_hidden = tf.contrib.layers.layer_norm(self.states_hidden)

			self.output = tf.contrib.layers.fully_connected(self.states_hidden,self.item_num,activation_fn=None)
			self.predict_prob = tf.nn.softmax(self.output)

			self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target,logits=self.output)
			self.loss = tf.reduce_mean(self.loss)
			self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

			# student loss
			if 'teacher' in self.name:
				pass
			else:
				self.soft_label = tf.placeholder(tf.float32, [None, self.item_num], name='soft_label')	# after softmax
				self.discriminator()

				self.stu_loss = -(tf.stop_gradient(self.soft_label) * tf.log(self.predict_prob)) + self.dis_loss
				# self.stu_loss = -(tf.stop_gradient(self.soft_label) * tf.log(self.predict_prob)) + tf.stop_gradient(self.dis_loss)
				self.stu_loss = tf.reduce_mean(self.stu_loss)

				# fix discriminator params
				train_var_list = [var for var in tf.trainable_variables() if (self.name in var.name) and ('discriminator' not in var.name)]
				self.stu_opt = tf.train.AdamOptimizer(self.learning_rate
					).minimize(self.stu_loss, var_list=train_var_list)

	def discriminator(self):
		self.hard_label = tf.placeholder(tf.float32, [None, 1], name='hard_label')	# {0, 1} -> student, teacher
		self.dis_input = self.predict_prob * 1.0

		discriminator = eval(self.args.discriminator_layers)
		discriminator.append(1)
		with tf.variable_scope("discriminator"):
			self.discriminator_output = mlp(self.dis_input, 
				self.is_training, hidden_sizes=discriminator, 
				dropout_rate=self.args.dis_dropout_rate, 
				output_activation=None,
				l2=tf.contrib.layers.l2_regularizer(self.args.weight_decay))

		self.dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminator_output, 
			labels=self.hard_label)
		self.dis_loss = tf.reduce_mean(self.dis_loss)
		train_dis_var_list = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
		self.dis_opt = tf.train.AdamOptimizer(self.dlr
			).minimize(self.dis_loss, var_list=train_dis_var_list)


class Caser:
	def __init__(self, args, hidden_size, learning_rate, dlr, item_num, state_size, embeddings, name='caser_teacher'):
		self.args = args
		self.state_size = state_size
		self.hw = state_size
		self.learning_rate = learning_rate
		self.dlr = dlr
		self.hidden_size = hidden_size
		self.item_num = int(item_num)
		self.name = name
		with tf.variable_scope(self.name):
			self.is_training = tf.placeholder(tf.bool, shape=())
			# all_embeddings = self.initialize_embeddings()
			all_embeddings = embeddings

			self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
			self.len_state=tf.placeholder(tf.int32, [None],name='len_state')
			self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss

			mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)

			self.input_emb=tf.nn.embedding_lookup(all_embeddings['state_embeddings'],self.inputs)
			self.input_emb*=mask
			self.embedded_chars_expanded = tf.expand_dims(self.input_emb, -1)

			# Create a convolution + maxpool layer for each filter size
			pooled_outputs = []
			num_filters=args.num_filters
			filter_sizes=eval(args.filter_sizes)
			for i, filter_size in enumerate(filter_sizes):
				with tf.name_scope("conv-maxpool-%s" % filter_size):
					# Convolution Layer
					filter_shape = [filter_size, self.hidden_size, 1, num_filters]
					W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
					b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

					conv = tf.nn.conv2d(
						self.embedded_chars_expanded,
						W,
						strides=[1, 1, 1, 1],
						padding="VALID",
						name="conv")
					# Apply nonlinearity
					h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
					# Maxpooling over the outputs
					# new shape after max_pool[?, 1, 1, num_filters]
					# be carefyul, the  new_sequence_length has changed because of wholesession[:, 0:-1]
					pooled = tf.nn.max_pool(
						h,
						ksize=[1, state_size - filter_size + 1, 1, 1],
						strides=[1, 1, 1, 1],
						padding='VALID',
						name="pool")
					pooled_outputs.append(pooled)

			# Combine all the pooled features
			num_filters_total = num_filters * len(filter_sizes)
			self.h_pool = tf.concat(pooled_outputs, 3)
			self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])  # shape=[batch_size, 384]
			# design the veritcal cnn
			with tf.name_scope("conv-verical"):
				filter_shape = [self.state_size, 1, 1, 1]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
				conv = tf.nn.conv2d(
					self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
			self.vcnn_flat = tf.reshape(h, [-1, self.hidden_size])
			self.final = tf.concat([self.h_pool_flat, self.vcnn_flat], 1)  # shape=[batch_size, 384+100]

			# Add dropout
			with tf.name_scope("dropout"):
				self.state_hidden = tf.layers.dropout(self.final,
										 rate=args.dropout_rate,
										 training=tf.convert_to_tensor(self.is_training))
			self.state_hidden=self.final    # shape=(?, 112)

			self.output = tf.contrib.layers.fully_connected(self.state_hidden,self.item_num,activation_fn=None)
			self.predict_prob = tf.nn.softmax(self.output)

			self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target,logits=self.output)
			self.loss = tf.reduce_mean(self.loss)
			self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

			# student loss
			if 'teacher' in self.name:
				pass
			else:
				self.soft_label = tf.placeholder(tf.float32, [None, self.item_num], name='soft_label')	# after softmax
				self.discriminator()

				self.stu_loss = -(tf.stop_gradient(self.soft_label) * tf.log(self.predict_prob)) + self.dis_loss
				# self.stu_loss = -(tf.stop_gradient(self.soft_label) * tf.log(self.predict_prob)) + tf.stop_gradient(self.dis_loss)
				self.stu_loss = tf.reduce_mean(self.stu_loss)

				# fix discriminator params
				train_var_list = [var for var in tf.trainable_variables() if (self.name in var.name) and ('discriminator' not in var.name)]
				self.stu_opt = tf.train.AdamOptimizer(self.learning_rate
					).minimize(self.stu_loss, var_list=train_var_list)

	def discriminator(self):
		self.hard_label = tf.placeholder(tf.float32, [None, 1], name='hard_label')	# {0, 1} -> student, teacher
		self.dis_input = self.predict_prob * 1.0

		discriminator = eval(self.args.discriminator_layers)
		discriminator.append(1)
		with tf.variable_scope("discriminator"):
			self.discriminator_output = mlp(self.dis_input, 
				self.is_training, hidden_sizes=discriminator, 
				dropout_rate=self.args.dis_dropout_rate, 
				output_activation=None,
				l2=tf.contrib.layers.l2_regularizer(self.args.weight_decay))

		self.dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminator_output, 
			labels=self.hard_label)
		self.dis_loss = tf.reduce_mean(self.dis_loss)
		train_dis_var_list = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
		self.dis_opt = tf.train.AdamOptimizer(self.dlr
			).minimize(self.dis_loss, var_list=train_dis_var_list)


class NextItNet:
	def __init__(self, args, hidden_size,learning_rate, dlr, item_num,state_size, embeddings, name='next_teacher'):
		self.args = args
		self.state_size = state_size
		self.hw = state_size
		self.learning_rate = learning_rate
		self.dlr = dlr
		self.hidden_size=hidden_size
		self.item_num=int(item_num)
		self.name = name
		with tf.variable_scope(self.name):
			self.is_training = tf.placeholder(tf.bool, shape=())
			# self.all_embeddings=self.initialize_embeddings()
			self.all_embeddings = embeddings

			self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
			self.len_state=tf.placeholder(tf.int32, [None],name='len_state')
			self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss
			mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)

			# self.input_emb=tf.nn.embedding_lookup(all_embeddings['state_embeddings'],self.inputs)
			self.model_para = {
				# 'dilated_channels': 64,  # larger is better until 512 or 1024
				'dilated_channels': hidden_size,
				'dilations': [1, 2, 1, 2, 1, 2, ],  # YOU should tune this hyper-parameter, refer to the paper.
				'kernel_size': 3,
			}

			context_embedding = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'],
													   self.inputs)
			context_embedding *= mask

			dilate_output = context_embedding
			for layer_id, dilation in enumerate(self.model_para['dilations']):
				dilate_output = nextitnet_residual_block(dilate_output, dilation,
														layer_id, self.model_para['dilated_channels'],
														self.model_para['kernel_size'], causal=True, train=self.is_training)
				dilate_output *= mask

			self.state_hidden = extract_axis_1(dilate_output, self.len_state - 1)

			self.output = tf.contrib.layers.fully_connected(self.state_hidden,self.item_num,activation_fn=None)
			self.predict_prob = tf.nn.softmax(self.output)

			self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target,logits=self.output)
			self.loss = tf.reduce_mean(self.loss)
			self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

			# student loss
			if 'teacher' in self.name:
				pass
			else:
				self.soft_label = tf.placeholder(tf.float32, [None, self.item_num], name='soft_label')	# after softmax
				self.discriminator()

				self.stu_loss = -(tf.stop_gradient(self.soft_label) * tf.log(self.predict_prob)) + self.dis_loss
				# self.stu_loss = -(tf.stop_gradient(self.soft_label) * tf.log(self.predict_prob)) + tf.stop_gradient(self.dis_loss)
				self.stu_loss = tf.reduce_mean(self.stu_loss)

				# fix discriminator params
				train_var_list = [var for var in tf.trainable_variables() if (self.name in var.name) and ('discriminator' not in var.name)]
				self.stu_opt = tf.train.AdamOptimizer(self.learning_rate
					).minimize(self.stu_loss, var_list=train_var_list)

	def discriminator(self):
		self.hard_label = tf.placeholder(tf.float32, [None, 1], name='hard_label')	# {0, 1} -> student, teacher
		self.dis_input = self.predict_prob * 1.0

		discriminator = eval(self.args.discriminator_layers)
		discriminator.append(1)
		with tf.variable_scope("discriminator"):
			self.discriminator_output = mlp(self.dis_input, 
				self.is_training, hidden_sizes=discriminator, 
				dropout_rate=self.args.dis_dropout_rate, 
				output_activation=None,
				l2=tf.contrib.layers.l2_regularizer(self.args.weight_decay))

		self.dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminator_output, 
			labels=self.hard_label)
		self.dis_loss = tf.reduce_mean(self.dis_loss)
		train_dis_var_list = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
		self.dis_opt = tf.train.AdamOptimizer(self.dlr
			).minimize(self.dis_loss, var_list=train_dis_var_list)


class SASRecnetwork:
	def __init__(self, args, hidden_size,learning_rate, dlr,item_num,state_size,embeddings,name='sas_teacher'):
		self.args = args
		self.state_size = state_size
		self.hw = state_size
		self.learning_rate = learning_rate
		self.dlr = dlr
		self.hidden_size = hidden_size
		self.item_num = int(item_num)
		self.name = name
		with tf.variable_scope(self.name):
			self.is_training = tf.placeholder(tf.bool, shape=())

			# all_embeddings=self.initialize_embeddings()
			all_embeddings = embeddings

			self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
			self.len_state=tf.placeholder(tf.int32, [None],name='len_state')
			self.target = tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss

			self.input_emb=tf.nn.embedding_lookup(all_embeddings['state_embeddings'],self.inputs)
			# Positional Encoding
			pos_emb=tf.nn.embedding_lookup(all_embeddings['pos_embeddings'],tf.tile(tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), 0), [tf.shape(self.inputs)[0], 1]))
			self.seq=self.input_emb+pos_emb

			mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)
			#Dropout
			self.seq = tf.layers.dropout(self.seq,
										 rate=args.dropout_rate,
										 training=tf.convert_to_tensor(self.is_training))
			self.seq *= mask

			# Build blocks

			for i in range(args.num_blocks):
				with tf.variable_scope("num_blocks_%d" % i):
					# Self-attention
					self.seq = multihead_attention(queries=normalize(self.seq),
												   keys=self.seq,
												   num_units=self.hidden_size,
												   num_heads=args.num_heads,
												   dropout_rate=args.dropout_rate,
												   is_training=self.is_training,
												   causality=True,
												   scope="self_attention")

					# Feed forward
					self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_size, self.hidden_size],
										   dropout_rate=args.dropout_rate,
										   is_training=self.is_training)
					self.seq *= mask

			self.seq = normalize(self.seq)
			self.state_hidden=extract_axis_1(self.seq, self.len_state - 1)

			self.output = tf.contrib.layers.fully_connected(self.state_hidden,self.item_num,activation_fn=None)
			self.predict_prob = tf.nn.softmax(self.output)

			self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target,logits=self.output)
			self.loss = tf.reduce_mean(self.loss)
			self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

			# student loss
			if 'teacher' in self.name:
				pass
			else:
				self.soft_label = tf.placeholder(tf.float32, [None, self.item_num], name='soft_label')	# after softmax
				self.discriminator()

				self.stu_loss = -(tf.stop_gradient(self.soft_label) * tf.log(self.predict_prob)) + self.dis_loss
				# self.stu_loss = -(tf.stop_gradient(self.soft_label) * tf.log(self.predict_prob)) + tf.stop_gradient(self.dis_loss)
				self.stu_loss = tf.reduce_mean(self.stu_loss)

				# fix discriminator params
				train_var_list = [var for var in tf.trainable_variables() if (self.name in var.name) and ('discriminator' not in var.name)]
				self.stu_opt = tf.train.AdamOptimizer(self.learning_rate
					).minimize(self.stu_loss, var_list=train_var_list)

	def discriminator(self):
		self.hard_label = tf.placeholder(tf.float32, [None, 1], name='hard_label')	# {0, 1} -> student, teacher
		self.dis_input = self.predict_prob * 1.0

		discriminator = eval(self.args.discriminator_layers)
		discriminator.append(1)
		with tf.variable_scope("discriminator"):
			self.discriminator_output = mlp(self.dis_input, 
				self.is_training, hidden_sizes=discriminator, 
				dropout_rate=self.args.dis_dropout_rate, 
				output_activation=None,
				l2=tf.contrib.layers.l2_regularizer(self.args.weight_decay))

		self.dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminator_output, 
			labels=self.hard_label)
		self.dis_loss = tf.reduce_mean(self.dis_loss)
		train_dis_var_list = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
		self.dis_opt = tf.train.AdamOptimizer(self.dlr
			).minimize(self.dis_loss, var_list=train_dis_var_list)


def initialize_embeddings(item_num, hidden_size, state_size=0, pos=False):
	all_embeddings = dict()
	state_embeddings= tf.Variable(tf.random_normal([item_num+1, hidden_size], 0.0, 0.01),
		name='state_embeddings')
	if pos:
		pos_embeddings=tf.Variable(tf.random_normal([state_size, hidden_size], 0.0, 0.01),
			name='pos_embeddings')
		all_embeddings['pos_embeddings'] = pos_embeddings

	all_embeddings['state_embeddings'] = state_embeddings
	return all_embeddings