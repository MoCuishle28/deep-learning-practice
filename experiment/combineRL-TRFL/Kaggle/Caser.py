import logging
import time
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import trfl
from utils import *

def parse_args():
	parser = argparse.ArgumentParser(description="Run supervised GRU.")
	parser.add_argument('--v', default='v')
	parser.add_argument('--base_log_dir', default='baseline-log/')
	parser.add_argument('--topk', default='5,10,20')
	parser.add_argument('--mode', default='valid')

	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--base_data_dir', nargs='?', default='../../data/kaggle-RL4REC', help='data directory')

	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--eval_interval', type=int, default=2000)
	parser.add_argument('--eval_batch', type=int, default=10)

	parser.add_argument('--hidden_factor', type=int, default=64,
						help='Number of hidden factors, i.e., embedding size.')

	parser.add_argument('--reward_buy', type=float, default=1.0)
	parser.add_argument('--reward_click', type=float, default=0.5)

	parser.add_argument('--lr', type=float, default=0.01,
						help='Learning rate.')
	parser.add_argument('--num_filters', type=int, default=16,
						help='Number of filters per filter size (default: 128)')
	parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
						help='Specify the filter_size')
	parser.add_argument('--dropout_rate', default=0.1, type=float)

	return parser.parse_args()


class Caser:
	def __init__(self, hidden_size, learning_rate, item_num, state_size):
		self.state_size = state_size
		self.hw = state_size
		self.learning_rate = learning_rate
		self.hidden_size = hidden_size
		self.item_num = int(item_num)
		self.is_training = tf.placeholder(tf.bool, shape=())
		all_embeddings = self.initialize_embeddings()

		self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
		# self.len_state=tf.placeholder(tf.int32, [None],name='len_state')
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

		self.output = tf.contrib.layers.fully_connected(self.state_hidden,self.item_num,activation_fn=None,scope='fc')

		self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target,logits=self.output)
		self.loss = tf.reduce_mean(self.loss)
		self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


	def initialize_embeddings(self):
		all_embeddings = dict()
		state_embeddings= tf.Variable(tf.random_normal([self.item_num+1, self.hidden_size], 0.0, 0.01),
			name='state_embeddings')
		all_embeddings['state_embeddings']=state_embeddings
		return all_embeddings

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


if __name__ == '__main__':
	# Network parameters
	args = parse_args()
	init_log(args)

	data_directory = args.base_data_dir
	data_statis = pd.read_pickle(
		os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
	state_size = data_statis['state_size'][0]  # the length of history to define the state
	item_num = data_statis['item_num'][0]  # total number of items
	# save_file = 'pretrain-GRU/%d' % (hidden_size)

	tf.reset_default_graph()

	CaserRec = Caser(hidden_size=args.hidden_factor, learning_rate=args.lr,item_num=item_num,state_size=state_size)

	replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
	saver = tf.train.Saver()

	total_step=0
	max_ndcg_and_epoch = [[0, 0, 0] for _ in args.topk.split(',')]	# (ng_click, ng_purchase, step)
	with tf.Session() as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())
		# evaluate(sess)
		num_rows=replay_buffer.shape[0]
		num_batches=int(num_rows/args.batch_size)
		for i in range(args.epoch):
			for j in range(num_batches):
				batch = replay_buffer.sample(n=args.batch_size).to_dict()
				state = list(batch['state'].values())
				# len_state = list(batch['len_state'].values())
				target=list(batch['action'].values())
				loss, _ = sess.run([CaserRec.loss, CaserRec.opt],
								   feed_dict={CaserRec.inputs: state,
											  CaserRec.target: target,
											  CaserRec.is_training:True})
				total_step += 1
				if (total_step == 1) or (total_step % 200 == 0):
					print("the loss in %dth batch is: %f" % (total_step, loss))
					logging.info("the loss in %dth batch is: %f" % (total_step, loss))
				if total_step % args.eval_interval == 0:
					evaluate(args, CaserRec, sess, max_ndcg_and_epoch, total_step, logging)
		# saver.save(sess, save_file)