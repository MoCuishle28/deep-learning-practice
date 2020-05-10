import os
import logging
import argparse
import time
import datetime
import random

import tensorflow as tf
import trfl
from trfl import indexing_ops
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from utils import *


class Ensemble(object):
	def __init__(self, args, name):
		super(Ensemble, self).__init__()
		self.args = args
		self.item_num = args.max_iid + 1
		self.hw = 10
		self.name = name
		with tf.variable_scope(self.name):
			self.inputs = tf.placeholder(tf.int32, [None, self.hw], name='inputs')
			self.len_state = tf.placeholder(tf.int32, [None], name='len_state')
			self.actions = tf.placeholder(tf.int32, [None], name='actions')

			self.all_embeddings = self.initialize_embeddings()
			self.item_emb = tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'], self.inputs)

			gru_out, self.states_hidden = tf.nn.dynamic_rnn(
				tf.contrib.rnn.GRUCell(self.args.seq_hidden_size), self.item_emb,
				dtype = tf.float32,
				sequence_length = self.len_state,
			)
			if args.layer_trick == 'ln':
				self.states_hidden = tf.contrib.layers.layer_norm(self.states_hidden)

			# ensemble models
			self.out0 = tf.contrib.layers.fully_connected(self.states_hidden, self.item_num, 
				activation_fn=None, 
				weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))
			self.out0_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, 
				logits=self.out0)
			self.out0_loss = tf.reduce_mean(self.out0_loss)
			self.out0_opt = tf.train.AdamOptimizer(self.args.lr).minimize(self.out0_loss)

			self.out1 = tf.contrib.layers.fully_connected(self.states_hidden, self.item_num, 
				activation_fn=None, 
				weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))
			self.out1_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, 
				logits=self.out1)
			self.out1_loss = tf.reduce_mean(self.out1_loss)
			self.out1_opt = tf.train.AdamOptimizer(self.args.lr).minimize(self.out1_loss)

			# get var logits
			logits_0 = indexing_ops.batched_index(self.out0, self.actions)	# (batch, 1/0 ?)
			logits_1 = indexing_ops.batched_index(self.out1, self.actions)
			# logits_0 = tf.reshape(logits_0, shape=(logits_0.shape[0], -1))
			# logits_1 = tf.reshape(logits_1, shape=(logits_1.shape[0], -1))
			logits_matrix = tf.stack([logits_0, logits_1], axis=1)		# (batch, ensemble num:2)
			_, self.var_logits = tf.nn.moments(logits_matrix, axes=1)
			# 在拿到后再计算 rewards
			# self.var_rewards = [1 if x <= args.q else -1 for x in var_logits.tolist()]

	def initialize_embeddings(self):
		all_embeddings = dict()
		item_embeddings = tf.Variable(tf.random_normal([self.args.max_iid + 2, self.args.i_emb_dim], 
			0.0, 0.01), name='item_embeddings')
		all_embeddings['item_embeddings'] = item_embeddings
		return all_embeddings



class DQN(object):
	def __init__(self, args, name='DQN'):
		super(DQN, self).__init__()
		self.args = args
		self.item_num = args.max_iid + 1
		self.hw = 10		# history window(过去多少次交互记录作为输入)
		# self.is_training = tf.placeholder(tf.bool, shape=())
		self.name = name
		with tf.variable_scope(self.name):
			self.discount = tf.placeholder(tf.float32, [None] , name="discount")
			self.rewards = tf.placeholder(tf.float32, [None], name='rewards')

			self.inputs = tf.placeholder(tf.int32, [None, self.hw], name='inputs')
			self.len_state = tf.placeholder(tf.int32, [None], name='len_state')

			self.actions = tf.placeholder(tf.int32, [None], name='actions')
			self.targetQ = tf.placeholder(tf.float32, [None, self.item_num], name='target')
			self.targetQ_selector = tf.placeholder(tf.float32, [None, self.item_num], 
				name='targetQ_selector')

			self.all_embeddings = self.initialize_embeddings()
			self.item_emb = tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'], self.inputs)

			gru_out, self.states_hidden = tf.nn.dynamic_rnn(
				tf.contrib.rnn.GRUCell(self.args.seq_hidden_size), self.item_emb,
				dtype = tf.float32,
				sequence_length = self.len_state,
			)
			if args.layer_trick == 'ln':
				self.states_hidden = tf.contrib.layers.layer_norm(self.states_hidden)

			# Q value
			self.output = tf.contrib.layers.fully_connected(self.states_hidden, self.item_num, 
				activation_fn=None, 
				weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))

			# BC loss
			self.bc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, 
				logits=self.output)
			self.bc_loss = tf.reduce_mean(self.bc_loss)
			self.bc_opt = tf.train.AdamOptimizer(self.args.lr).minimize(self.bc_loss)

			# RL loss
			loss, q_learning = trfl.double_qlearning(self.output, self.actions, self.rewards, 
				self.discount, self.targetQ, self.targetQ_selector)

			self.rl_loss = tf.reduce_mean(loss)
			self.rl_optim = tf.train.AdamOptimizer(self.args.lr).minimize(self.rl_loss)

	def initialize_embeddings(self):
		all_embeddings = dict()
		item_embeddings = tf.Variable(tf.random_normal([self.args.max_iid + 2, self.args.i_emb_dim], 
			0.0, 0.01), name='item_embeddings')
		all_embeddings['item_embeddings'] = item_embeddings
		return all_embeddings

	def get_qnetwork_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


class Run(object):
	def __init__(self, args, mainQ, targetQ, ensemble, sess, update_target, copy_weight):
		super(Run, self).__init__()
		self.args = args
		self.mainQ = mainQ
		self.targetQ = targetQ
		self.target_network_update_ops = update_target
		self.copy_weight = copy_weight
		self.ensemble = ensemble
		self.sess = sess

		self.topk = [int(x) for x in args.topk.split(',')]
		self.replay_buffer = pd.read_pickle(os.path.join(args.base_data_dir, 'replay_buffer.df'))

	def sample_data(self):
		batch = self.replay_buffer.sample(n=args.batch_size).to_dict()
		state = list(batch['state'].values())
		len_state = list(batch['len_state'].values())
		next_state = list(batch['next_state'].values())
		len_next_states = list(batch['len_next_states'].values())
		target_items = list(batch['action'].values())
		is_done = list(batch['is_done'].values())
		return state, len_state, next_state, len_next_states, target_items, is_done

	def get_var_rewards(self, state, len_state, target_items):
		var_logits = self.sess.run(self.ensemble.var_logits,
		   feed_dict={self.ensemble.inputs: state,
					  self.ensemble.len_state: len_state,
					  self.ensemble.actions: target_items})
		var_rewards = [1 if x <= self.args.q else -1 for x in var_logits.tolist()]
		return var_rewards

	def train_agent(self):
		num_rows = self.replay_buffer.shape[0]
		num_batches = int(num_rows / self.args.batch_size)
		max_ndcg_and_epoch = [[0, 0, 0] for _ in self.args.topk.split(',')]	# (ng_click, ng_purchase, step)
		total_step = 0

		self.sess.run(self.copy_weight)		# copy weights
		discount = [self.args.gamma] * self.args.batch_size
		for i_epoch in range(self.args.epoch):
			for j in range(num_batches):
				state, len_state, next_state, len_next_states, target_items, is_done = self.sample_data()
				current_actions = self.sess.run(self.mainQ.output, feed_dict={
					self.mainQ.inputs: state,
					self.mainQ.len_state: len_state})
				current_actions = np.argmax(current_actions, 1)
				# get target state-action value
				target = self.sess.run(self.targetQ.output, feed_dict={self.targetQ.inputs: next_state,
					self.targetQ.len_state: len_next_states})
				# get selector
				targetQ_selector = self.sess.run(self.mainQ.output, feed_dict={self.mainQ.inputs: next_state,
					self.mainQ.len_state:len_next_states})

				for index in range(args.batch_size):
					if is_done[index]:
						target[index] = np.zeros([self.mainQ.item_num])

				bc_loss, _ = self.sess.run([self.mainQ.bc_loss, self.mainQ.bc_opt],
					   feed_dict={self.mainQ.inputs: state,
								  self.mainQ.len_state: len_state,
								  self.mainQ.actions: target_items})
				rewards = self.get_var_rewards(state, len_state, current_actions)
				rl_loss, _ = self.sess.run([self.mainQ.rl_loss, self.mainQ.rl_optim],
								   feed_dict={self.mainQ.inputs: state,
											  self.mainQ.len_state: len_state,
											  self.mainQ.actions: target_items,
											  self.mainQ.targetQ: target,
											  self.mainQ.targetQ_selector: targetQ_selector,
											  self.mainQ.discount: discount,
											  self.mainQ.rewards: rewards})

				self.sess.run(self.target_network_update_ops)		# update target net
				total_step += 1
				if (total_step == 1) or (total_step % 200 == 0):
					rl_loss, bc_loss = round(rl_loss.item(), 5), round(bc_loss.item(), 5)
					info = f"[Agent] epoch:{i_epoch} Step:{total_step}, BC loss:{bc_loss}, RL loss:{rl_loss}"
					print(info)
					logging.info(info)
				if total_step % args.eval_interval == 0:
					t1 = time.time()
					evaluate(args, self.mainQ, self.sess, max_ndcg_and_epoch, total_step, logging)
					t2 = time.time()
					print(f'Time:{t2 - t1}')
					logging.info(f'Time:{t2 - t1}')


	def train_ensemble(self):
		num_rows = self.replay_buffer.shape[0]
		num_batches = int(num_rows / self.args.batch_size)
		total_step = 0

		for i_epoch in range(self.args.e_epoch):
			for j in range(num_batches):
				batch = self.replay_buffer.sample(n=args.batch_size).to_dict()
				state = list(batch['state'].values())
				len_state = list(batch['len_state'].values())	
				target_items = list(batch['action'].values())

				loss_0, _ = self.sess.run([self.ensemble.out0_loss, self.ensemble.out0_opt],
					   feed_dict={self.ensemble.inputs: state,
								  self.ensemble.len_state: len_state,
								  self.ensemble.actions: target_items})

				loss_1, _ = self.sess.run([self.ensemble.out1_loss, self.ensemble.out1_opt],
					   feed_dict={self.ensemble.inputs: state,
								  self.ensemble.len_state: len_state,
								  self.ensemble.actions: target_items})
				total_step += 1
				if (total_step == 1) or (total_step % 200 == 0):
					loss_0, loss_1 = round(loss_0.item(), 4), round(loss_1.item(), 4)
					info = f"[Ensemble] epoch:{i_epoch} Step:{total_step}, loss_0:{loss_0}, loss_1:{loss_1}"
					print(info)
					logging.info(info)


def main(args):
	tf.reset_default_graph()
	trainQ = DQN(args, name='trainQ')
	targetQ = DQN(args, name='targetQ')
	ensemble = Ensemble(args, name='ensemble')

	target_network_update_ops = trfl.update_target_variables(trainQ.get_qnetwork_variables(), 
		targetQ.get_qnetwork_variables(), tau=args.tau)
	copy_weight = trfl.update_target_variables(trainQ.get_qnetwork_variables(), 
		targetQ.get_qnetwork_variables(), tau=1.0)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.graph.finalize()

		run = Run(args, trainQ, targetQ, ensemble, sess, target_network_update_ops, copy_weight)
		run.train_ensemble()
		run.train_agent()


def init_log(args):
	if not os.path.exists(args.base_log_dir):
		os.makedirs(args.base_log_dir)
	if not os.path.exists(args.base_pic_dir):
		os.makedirs(args.base_pic_dir)
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
	base_data_dir = '../data/'
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--v', default="v")
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_pic_dir', default="pic/")
	parser.add_argument('--base_data_dir', default=base_data_dir + 'kaggle-RL4REC')
	parser.add_argument('--show', default='n')
	parser.add_argument('--mode', default='valid')		# test/valid
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--eval_interval', type=int, default=6000)
	parser.add_argument('--eval_batch', type=int, default=10)

	parser.add_argument('--e_epoch', type=int, default=100)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--topk', default='5,10,20')
	parser.add_argument('--batch_size', type=int, default=256)

	parser.add_argument('--optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	# embedding
	parser.add_argument('--max_iid', type=int, default=70851)	# 0~70851
	parser.add_argument('--i_emb_dim', type=int, default=64)
	# double DQN
	parser.add_argument('--reward_buy', type=float, default=1.0)
	parser.add_argument('--reward_click', type=float, default=0.5)
	parser.add_argument('--q', type=float, default=0.98)			# threshold

	parser.add_argument('--seq_hidden_size', type=int, default=64)
	parser.add_argument('--seq_layer_num', type=int, default=1)

	parser.add_argument('--tau', type=float, default=0.001)
	parser.add_argument('--gamma', type=float, default=0.5)
	parser.add_argument('--layer_trick', default='ln')			# ln/bn/none
	parser.add_argument('--dropout', type=float, default=0.0)
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	init_log(args)
	main(args)