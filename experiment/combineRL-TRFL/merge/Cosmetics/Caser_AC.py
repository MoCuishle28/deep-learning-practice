import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import trfl
from utils import *
from trfl import indexing_ops
import logging
import time
import datetime

def init_log(args):
	if not os.path.exists(args.base_log_dir):
		os.makedirs(args.base_log_dir)
	start = datetime.datetime.now()
	logging.basicConfig(level = logging.INFO,
					filename = args.base_log_dir + args.v + '-' + str(time.time()) + '.log',
					filemode = 'a',
					)
	print('start! '+str(start))
	logging.info('start! '+str(start))
	logging.info('Parameter:')
	logging.info(str(args))
	logging.info('\n-------------------------------------------------------------\n')


def parse_args():
	parser = argparse.ArgumentParser(description="Run supervised Caser_AC.")
	parser.add_argument('--mode', default='valid')
	parser.add_argument('--v', default='v')
	parser.add_argument('--topk', default='5,10,20')
	parser.add_argument('--eval_batch', type=int, default=10)
	parser.add_argument('--mem_ratio', type=float, default=0.2)
	parser.add_argument('--cuda', default='0')

	parser.add_argument('--epoch', type=int, default=30,
						help='Number of max epochs.')
	parser.add_argument('--base_log_dir', default='baseline-log/')
	parser.add_argument('--base_data_dir', default='RC19')
	# parser.add_argument('--pretrain', type=int, default=1,
	#                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
	parser.add_argument('--batch_size', type=int, default=256,
						help='Batch size.')
	parser.add_argument('--hidden_factor', type=int, default=64,
						help='Number of hidden factors, i.e., embedding size.')

	parser.add_argument('--r', type=float, default=1.0, help='reward.')

	parser.add_argument('--lr', type=float, default=0.01,
						help='Learning rate.')
	parser.add_argument('--num_filters', type=int, default=16,
						help='Number of filters per filter size (default: 128)')
	parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
						help='Specify the filter_size')
	parser.add_argument('--discount', type=float, default=0.5,
						help='Discount factor for RL.')
	parser.add_argument('--dropout_rate', default=0.1, type=float)
	return parser.parse_args()


class Caser:
	def __init__(self, hidden_size,learning_rate,item_num,state_size,name='CaserRec'):
		self.state_size = state_size
		self.hw = state_size
		self.learning_rate = learning_rate
		self.hidden_size=hidden_size
		self.item_num=int(item_num)
		self.is_training = tf.placeholder(tf.bool, shape=())
		self.name=name
		with tf.variable_scope(self.name):
			self.all_embeddings=self.initialize_embeddings()

			self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
			self.len_state=tf.placeholder(tf.int32, [None],name='len_state')
			self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss

			mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)

			self.input_emb = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'], self.inputs)
			self.input_emb *= mask
			self.embedded_chars_expanded = tf.expand_dims(self.input_emb, -1)

			# Create a convolution + maxpool layer for each filter size
			pooled_outputs = []
			num_filters = args.num_filters
			filter_sizes = eval(args.filter_sizes)
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
			self.state_hidden = self.final

			self.output1 = tf.contrib.layers.fully_connected(self.state_hidden, self.item_num,
															 activation_fn=None, scope="q-value")  # all q-values
			self.output2 = tf.contrib.layers.fully_connected(self.state_hidden, self.item_num,
															 activation_fn=None, scope="ce-logits")  # all ce logits

			# TRFL way
			self.actions = tf.placeholder(tf.int32, [None])
			self.targetQs_ = tf.placeholder(tf.float32, [None, item_num])
			self.targetQs_selector = tf.placeholder(tf.float32, [None,
																 item_num])  # used for select best action for double q learning
			self.reward = tf.placeholder(tf.float32, [None])
			self.discount = tf.placeholder(tf.float32, [None])

			# TRFL double qlearning
			qloss, q_learning = trfl.double_qlearning(self.output1, self.actions, self.reward, self.discount,
													  self.targetQs_, self.targetQs_selector)
			q_indexed = tf.stop_gradient(indexing_ops.batched_index(self.output1, self.actions))
			celoss = tf.multiply(q_indexed, tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions,
																						   logits=self.output2))
			self.loss = tf.reduce_mean(qloss + celoss)
			self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


	def initialize_embeddings(self):
		all_embeddings = dict()
		state_embeddings= tf.Variable(tf.random_normal([self.item_num+1, self.hidden_size], 0.0, 0.01),
			name='state_embeddings')
		all_embeddings['state_embeddings']=state_embeddings
		return all_embeddings


if __name__ == '__main__':
	# Network parameters
	args = parse_args()
	init_log(args)
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

	base_data_dir = '../../../data/'
	args.base_data_dir = os.path.join(base_data_dir, args.base_data_dir)

	data_directory = args.base_data_dir
	data_statis = pd.read_pickle(
		os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
	state_size = data_statis['state_size'][0]  # the length of history to define the state
	item_num = data_statis['item_num'][0]  # total number of items
	# save_file = 'pretrain-GRU/%d' % (hidden_size)

	tf.reset_default_graph()

	CaserRec1 = Caser(hidden_size=args.hidden_factor, learning_rate=args.lr,item_num=item_num,state_size=state_size,name='CaserRec1')
	CaserRec2 = Caser(hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num, state_size=state_size,
					  name='CaserRec2')

	replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
	saver = tf.train.Saver()

	total_step=0
	max_ndcg_and_epoch = [[0, 0] for _ in args.topk.split(',')]	# (ng_inter, step)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.mem_ratio)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())
		# evaluate(sess)
		num_rows=replay_buffer.shape[0]
		num_batches=int(num_rows/args.batch_size)
		for i in range(args.epoch):
			for j in range(num_batches):
				batch = replay_buffer.sample(n=args.batch_size).to_dict()
				next_state = list(batch['next_state'].values())
				len_next_state = list(batch['len_next_states'].values())
				# double q learning, pointer is for selecting which network  is target and which is main
				pointer = np.random.randint(0, 2)
				if pointer == 0:
					mainQN = CaserRec1
					target_QN = CaserRec2
				else:
					mainQN = CaserRec2
					target_QN = CaserRec1
				target_Qs = sess.run(target_QN.output1,
									 feed_dict={target_QN.inputs: next_state,
												target_QN.len_state: len_next_state,
												target_QN.is_training: True})
				target_Qs_selector = sess.run(mainQN.output1,
											  feed_dict={mainQN.inputs: next_state,
														 mainQN.len_state: len_next_state,
														 mainQN.is_training: True})

				# Set target_Qs to 0 for states where episode ends
				is_done = list(batch['is_done'].values())
				for index in range(target_Qs.shape[0]):
					if is_done[index]:
						target_Qs[index] = np.zeros([item_num])

				state = list(batch['state'].values())
				len_state = list(batch['len_state'].values())
				action = list(batch['action'].values())
				reward = []
				for k in range(len(action)):
					reward.append(args.r)
				discount = [args.discount] * len(action)

				loss, _ = sess.run([mainQN.loss, mainQN.opt],
								   feed_dict={mainQN.inputs: state,
											  mainQN.len_state: len_state,
											  mainQN.targetQs_: target_Qs,
											  mainQN.reward: reward,
											  mainQN.discount: discount,
											  mainQN.actions: action,
											  mainQN.targetQs_selector: target_Qs_selector,
											  mainQN.is_training: True})
				total_step += 1
				if total_step % 200 == 0:
					print("the loss in %dth batch is: %f" % (total_step, loss))
					logging.info("the loss in %dth batch is: %f" % (total_step, loss))
				if total_step % 2000 == 0:
					evaluate(args, CaserRec1, sess, max_ndcg_and_epoch, total_step, logging, RL=True)