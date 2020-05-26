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


class Agent(object):
	def __init__(self, args, name='Agent', dqda_clipping=None, clip_norm=False, max_action=1.0):
		super(Agent, self).__init__()
		self.args = args
		self.hw = 10		# history window(过去多少次交互记录作为输入)
		self.item_num = args.max_iid + 1
		# self.is_training = tf.placeholder(tf.bool, shape=())
		self.name = name
		self.is_training = tf.placeholder(tf.bool, shape=())
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
				tf.contrib.rnn.GRUCell(self.args.hidden_factor), self.item_emb,
				dtype = tf.float32,
				sequence_length = self.len_state,
			)
			if args.layer_trick == 'ln':
				self.states_hidden = tf.contrib.layers.layer_norm(self.states_hidden)

			self.logits = tf.contrib.layers.fully_connected(self.states_hidden, 
				args.max_iid + 1, 
				activation_fn=None, 
				weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))

			# DQN loss
			loss, q_learning = trfl.double_qlearning(self.logits, self.actions, self.rewards, 
				self.discount, self.targetQ, self.targetQ_selector)
			self.rl_loss = tf.reduce_mean(loss)

			# supervised learning
			self.sl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, 
				logits=self.logits)
			self.sl_loss = tf.reduce_mean(self.sl_loss)

			self.loss = self.rl_loss + args.lambda0 * self.sl_loss
			self.train = tf.train.AdamOptimizer(args.lr).minimize(self.loss)

	def initialize_embeddings(self):
		all_embeddings = dict()
		item_embeddings = tf.Variable(tf.random_normal([self.args.max_iid + 2, self.args.i_emb_dim], 
			0.0, 0.01), name='item_embeddings')
		all_embeddings['item_embeddings'] = item_embeddings
		return all_embeddings

	def get_qnetwork_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


class Run(object):
	def __init__(self, args, main_agent, target_agent):
		super(Run, self).__init__()
		self.args = args
		self.mainQ = main_agent
		self.targetQ = target_agent
		self.target_network_update_ops = trfl.update_target_variables(target_agent.get_qnetwork_variables(), 
			main_agent.get_qnetwork_variables(), tau=args.tau)
		self.copy_weight = trfl.update_target_variables(target_agent.get_qnetwork_variables(), 
			main_agent.get_qnetwork_variables(), tau=1.0)

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
		is_buy = list(batch['is_buy'].values())
		return state, len_state, next_state, len_next_states, target_items, is_done, is_buy

	def cal_rewards(self, is_buy):
		rewards = []
		for k in range(len(is_buy)):
			rewards.append(self.args.reward_buy if is_buy[k] == 1 else self.args.reward_click)
		return rewards

	def train(self):
		num_rows = self.replay_buffer.shape[0]
		num_batches = int(num_rows / self.args.batch_size)
		max_ndcg_and_epoch = [[0, 0, 0] for _ in self.args.topk.split(',')]	# (ng_click, ng_purchase, step)
		total_step = 0

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.args.mem_ratio)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			sess.run(tf.global_variables_initializer())
			sess.graph.finalize()
			sess.run(self.copy_weight)		# copy weights
			discount = [self.args.gamma] * self.args.batch_size
			for i_epoch in range(self.args.epoch):
				for j in range(num_batches):
					state, len_state, next_state, len_next_states, target_items, is_done, is_buy = self.sample_data()
					# get target Q value
					target = sess.run(self.targetQ.logits, feed_dict={
						self.targetQ.inputs: next_state,
						self.targetQ.len_state: len_next_states})
					# get selector
					targetQ_selector = sess.run(self.mainQ.logits, feed_dict={
						self.mainQ.inputs: next_state,
						self.mainQ.len_state:len_next_states})

					for index in range(args.batch_size):
						if is_done[index]:
							target[index] = np.zeros([self.mainQ.item_num])

					rewards = self.cal_rewards(is_buy)
					rl_loss, sl_loss, _ = sess.run([self.mainQ.rl_loss, self.mainQ.sl_loss, self.mainQ.train],
									   feed_dict={self.mainQ.inputs: state,
												  self.mainQ.len_state: len_state,
												  self.mainQ.actions: target_items,
												  self.mainQ.targetQ: target,
												  self.mainQ.targetQ_selector: targetQ_selector,
												  self.mainQ.discount: discount,
												  self.mainQ.rewards: rewards})

					sess.run(self.target_network_update_ops)		# update target net
					total_step += 1
					if (total_step == 1) or (total_step % 200 == 0):
						mean_r = round(np.array(rewards).mean().item(), 4)
						rl_loss, sl_loss = round(rl_loss.item(), 4), round(sl_loss.item(), 4)
						info = f"[{i_epoch+1}/{self.args.epoch}] Step:{total_step}, mean reward:{mean_r}, SL loss:{sl_loss}, RL loss:{rl_loss}"
						print(info)
						logging.info(info)
					if total_step % args.eval_interval == 0:
						t1 = time.time()
						evaluate_multi_head(args, self.mainQ, sess, max_ndcg_and_epoch, total_step, logging)
						t2 = time.time()
						print(f'Time:{t2 - t1}')
						logging.info(f'Time:{t2 - t1}')


def main(args):
	tf.reset_default_graph()
	main_agent = Agent(args, name='main')
	target_agent = Agent(args, name='target')

	run = Run(args, main_agent, target_agent)
	run.train()


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
	base_data_dir = '../../data/'
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--v', default="v")
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_pic_dir', default="pic/")
	parser.add_argument('--base_data_dir', default=base_data_dir + 'RC15')
	parser.add_argument('--mode', default='valid')		# test/valid
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--eval_interval', type=int, default=1000)
	parser.add_argument('--eval_batch', type=int, default=10)
	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--topk', default='5,10,20')

	parser.add_argument('--weight_decay', type=float, default=1e-4)
	# embedding
	parser.add_argument('--max_iid', type=int, default=26702)	# 0~26702
	parser.add_argument('--i_emb_dim', type=int, default=64)

	parser.add_argument('--reward_buy', type=float, default=1.0)
	parser.add_argument('--reward_click', type=float, default=0.2)

	parser.add_argument('--hidden_factor', type=int, default=64)
	parser.add_argument('--lambda0', type=float, default=1.0)
	parser.add_argument('--lr', type=float, default=5e-3)

	parser.add_argument('--tau', type=float, default=0.001)
	parser.add_argument('--gamma', type=float, default=0.5)
	parser.add_argument('--layer_trick', default='ln')			# ln/bn/none
	parser.add_argument('--dropout', type=float, default=1.0)
	parser.add_argument('--mem_ratio', type=float, default=0.2)
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	init_log(args)
	main(args)