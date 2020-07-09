import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import trfl
from utils import *
from SASRecModules import *
import logging
import time
import datetime

def parse_args():
	parser = argparse.ArgumentParser(description="Run supervised GRU.")
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
	parser.add_argument('--batch_size', type=int, default=256,
						help='Batch size.')
	parser.add_argument('--hidden_factor', type=int, default=64,
						help='Number of hidden factors, i.e., embedding size.')
	parser.add_argument('--r', type=float, default=1.0, help='reward.')
	parser.add_argument('--lr', type=float, default=0.001,
						help='Learning rate.')
	parser.add_argument('--num_heads', default=1, type=int)
	parser.add_argument('--num_blocks', default=1, type=int)
	parser.add_argument('--discount', type=float, default=0.5,
						help='Discount factor for RL.')
	parser.add_argument('--dropout_rate', default=0.1, type=float)
	return parser.parse_args()


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


class SASRecnetwork:
	def __init__(self, hidden_size,learning_rate,item_num,state_size, name='DQNetwork'):
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

			self.input_emb=tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'],self.inputs)
			# Positional Encoding
			pos_emb=tf.nn.embedding_lookup(self.all_embeddings['pos_embeddings'],tf.tile(tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), 0), [tf.shape(self.inputs)[0], 1]))
			self.seq=self.input_emb+pos_emb

			mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)
			# Dropout
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

			self.output1 = tf.contrib.layers.fully_connected(self.state_hidden, self.item_num,
															 activation_fn=None, scope="q-value")  # all q-values
			self.output2 = tf.contrib.layers.fully_connected(self.state_hidden, self.item_num,
															 activation_fn=None, scope="ce-logits")  # all logits

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
			celoss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2)
			self.loss = tf.reduce_mean(celoss+qloss)
			self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


	def initialize_embeddings(self):
		all_embeddings = dict()
		state_embeddings= tf.Variable(tf.random_normal([self.item_num+1, self.hidden_size], 0.0, 0.01),
			name='state_embeddings')
		pos_embeddings=tf.Variable(tf.random_normal([self.state_size, self.hidden_size], 0.0, 0.01),
			name='pos_embeddings')
		all_embeddings['state_embeddings']=state_embeddings
		all_embeddings['pos_embeddings']=pos_embeddings
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

	SASRec1 = SASRecnetwork(hidden_size=args.hidden_factor, learning_rate=args.lr,item_num=item_num,state_size=state_size,name='SASRec1')
	SASRec2 = SASRecnetwork(hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
							state_size=state_size, name='SASRec2')

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
					mainQN = SASRec1
					target_QN = SASRec2
				else:
					mainQN = SASRec2
					target_QN = SASRec1
				target_Qs = sess.run(target_QN.output1,
									 feed_dict={target_QN.inputs: next_state,
												target_QN.len_state: len_next_state,
												target_QN.is_training:True})
				target_Qs_selector = sess.run(mainQN.output1,
											  feed_dict={mainQN.inputs: next_state,
														 mainQN.len_state: len_next_state,
														 mainQN.is_training:True})

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
											  mainQN.is_training:True})
				total_step += 1
				if total_step % 200 == 0:
					print("the loss in %dth batch is: %f" % (total_step, loss))
					logging.info("the loss in %dth batch is: %f" % (total_step, loss))
				if total_step % 2000== 0:
					evaluate(args, SASRec1, sess, max_ndcg_and_epoch, total_step, logging, RL=True)