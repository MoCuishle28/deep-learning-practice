import logging
import time
import datetime
# --eval_interval 1 --epoch 1

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import trfl
from utils import *
from NextItNetModules import *


def parse_args():
	parser = argparse.ArgumentParser(description="Run supervised NextItNet.")

	parser.add_argument('--v', default='v')
	parser.add_argument('--base_log_dir', default='baseline-log/')
	parser.add_argument('--topk', default='5,10,20')
	parser.add_argument('--mode', default='valid')

	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--base_data_dir', nargs='?', default='../../../data/kaggle-RL4REC', help='data directory')

	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--eval_interval', type=int, default=2000)
	parser.add_argument('--eval_batch', type=int, default=10)

	parser.add_argument('--hidden_factor', type=int, default=64,
						help='Number of hidden factors, i.e., embedding size.')

	parser.add_argument('--reward_buy', type=float, default=1.0)
	parser.add_argument('--reward_click', type=float, default=0.5)

	parser.add_argument('--lr', type=float, default=0.005,
						help='Learning rate.')
	parser.add_argument('--mem_ratio', type=float, default=0.2)
	parser.add_argument('--cuda', default='0')
	return parser.parse_args()


class NextItNet:
	def __init__(self, hidden_size,learning_rate,item_num,state_size):
		self.state_size = state_size
		self.hw = state_size
		self.learning_rate = learning_rate
		self.hidden_size=hidden_size
		self.item_num=int(item_num)
		self.is_training = tf.placeholder(tf.bool, shape=())
		self.all_embeddings=self.initialize_embeddings()

		self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
		self.len_state=tf.placeholder(tf.int32, [None],name='len_state')
		self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss
		mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)

		# self.input_emb=tf.nn.embedding_lookup(all_embeddings['state_embeddings'],self.inputs)
		self.model_para = {
			'dilated_channels': 64,  # larger is better until 512 or 1024
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

		self.output = tf.contrib.layers.fully_connected(self.state_hidden,self.item_num,activation_fn=None,scope='fc',weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

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
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	init_log(args)

	data_directory = args.base_data_dir
	data_statis = pd.read_pickle(
		os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
	state_size = data_statis['state_size'][0]  # the length of history to define the state
	item_num = data_statis['item_num'][0]  # total number of items

	tf.reset_default_graph()

	NextRec = NextItNet(hidden_size=args.hidden_factor, learning_rate=args.lr,item_num=item_num,state_size=state_size)

	replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))

	total_step = 0
	max_ndcg_and_epoch = [[0, 0] for _ in args.topk.split(',')]	# (ng_inter, step)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.mem_ratio)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())
		# evaluate(sess)
		num_rows = replay_buffer.shape[0]
		num_batches = int(num_rows/args.batch_size)
		for i in range(args.epoch):
			for j in range(num_batches):
				batch = replay_buffer.sample(n=args.batch_size).to_dict()
				state = list(batch['state'].values())
				len_state = list(batch['len_state'].values())
				target=list(batch['action'].values())
				loss, _ = sess.run([NextRec.loss, NextRec.opt],
								   feed_dict={NextRec.inputs: state,
											  NextRec.len_state: len_state,
											  NextRec.target: target,
											  NextRec.is_training:True})
				total_step += 1
				if total_step % 200 == 0:
					print("the loss in %dth batch is: %f" % (total_step, loss))
					logging.info("the loss in %dth batch is: %f" % (total_step, loss))
				if total_step % args.eval_interval == 0:
					evaluate(args, NextRec, sess, max_ndcg_and_epoch, total_step, logging)