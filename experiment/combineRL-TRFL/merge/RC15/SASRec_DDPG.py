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
from SASRecModules import *


class Agent:
	def __init__(self, args, name='Agent', dqda_clipping=None, clip_norm=False, max_action=1.0):
		self.args = args
		self.name = name
		self.hw = 10
		self.hidden_size = args.hidden_factor
		self.item_num = args.max_iid + 1

		with tf.variable_scope(self.name):
			self.is_training = tf.placeholder(tf.bool, shape=())

			all_embeddings = self.initialize_embeddings()

			self.inputs = tf.placeholder(tf.int32, [None, self.hw],name='inputs')
			self.len_state = tf.placeholder(tf.int32, [None],name='len_state')
			self.target_items = tf.placeholder(tf.int32, [None], name='target_items')

			self.discount = tf.placeholder(tf.float32, [None] , name="discount")
			self.reward = tf.placeholder(tf.float32, [None], name='reward')
			self.target = tf.placeholder(tf.float32, [None],name='target')

			self.input_emb = tf.nn.embedding_lookup(all_embeddings['state_embeddings'],self.inputs)
			# Positional Encoding
			pos_emb = tf.nn.embedding_lookup(all_embeddings['pos_embeddings'],tf.tile(tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), 0), [tf.shape(self.inputs)[0], 1]))
			self.seq = self.input_emb+pos_emb

			mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, self.item_num)), -1)
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
			self.state_hidden = extract_axis_1(self.seq, self.len_state - 1)
			self.action_size = int(self.state_hidden.shape[-1])

			# DDPG
			self.actor_output = tf.contrib.layers.fully_connected(self.state_hidden, self.action_size, 
					activation_fn=tf.nn.tanh, 
					weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))
			self.actor_out_ = self.actor_output * max_action

			self.critic_input = tf.concat([self.actor_out_, self.state_hidden], axis=1)
			self.critic_output = tf.contrib.layers.fully_connected(self.critic_input, 1, 
				activation_fn=None, 
				weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))

			self.dpg_return = trfl.dpg(self.critic_output, self.actor_out_, 
				dqda_clipping=dqda_clipping, clip_norm=clip_norm)

			self.actor_loss = tf.reduce_mean(self.dpg_return.loss)
			self.actor_optim = tf.train.AdamOptimizer(args.alr).minimize(self.actor_loss)

			self.td_return = trfl.td_learning(tf.squeeze(self.critic_output), self.reward, 
				self.discount, self.target)
			self.critic_loss = tf.reduce_mean(self.td_return.loss)
			self.critic_optim = tf.train.AdamOptimizer(args.clr).minimize(self.critic_loss)

			# SASRec
			self.ranking_model_input = self.actor_out_ + self.state_hidden
			self.logits = tf.contrib.layers.fully_connected(self.ranking_model_input, self.item_num,
				activation_fn=None, weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))

			self.ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_items,
				logits=self.logits)
			self.ranking_model_loss = tf.reduce_mean(self.ce_loss)
			self.model_optim = tf.train.AdamOptimizer(args.mlr).minimize(self.ranking_model_loss)


	def initialize_embeddings(self):
		all_embeddings = dict()
		state_embeddings= tf.Variable(tf.random_normal([self.item_num+1, self.hidden_size], 0.0, 0.01),
			name='state_embeddings')
		pos_embeddings=tf.Variable(tf.random_normal([self.hw, self.hidden_size], 0.0, 0.01),
			name='pos_embeddings')
		all_embeddings['state_embeddings']=state_embeddings
		all_embeddings['pos_embeddings']=pos_embeddings
		return all_embeddings

	def get_qnetwork_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


class Run(object):
	def __init__(self, args, main_agent, target_agent):
		super(Run, self).__init__()
		self.args = args
		self.main_agent = main_agent
		self.target_agent = target_agent
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
		return state, len_state, next_state, len_next_states, target_items, is_done

	def cal_rewards(self, logits, target_items):
		logits = torch.tensor(logits)
		_, rankings = logits.topk(self.args.reward_top)
		rankings = rankings.tolist()	# (batch, topk)
		rewards = []
		for target_iid, rec_list in zip(target_items, rankings):
			ndcg = 0.0
			for i, iid in enumerate(rec_list):
				if iid == target_iid:
					ndcg = 1.0 / np.log2(i + 2.0).item()
					break
			rewards.append(ndcg)
		return rewards

	def state_trans(self, rewards, state, next_state, len_state, len_next_states):
		true_next_state, true_next_state_len = [], []
		for r, s, s_, sl, sl_ in zip(rewards, state, next_state, len_state, len_next_states):
			if r == 0:
				true_next_state.append(s)
				true_next_state_len.append(sl)
			else:
				true_next_state.append(s_)
				true_next_state_len.append(sl_)
		return true_next_state, true_next_state_len

	def train(self):
		num_rows = self.replay_buffer.shape[0]
		num_batches = int(num_rows / self.args.batch_size)
		max_ndcg_and_epoch = [[0, 0] for _ in args.topk.split(',')]	# (ng_inter, step)
		total_step = 0

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.args.mem_ratio)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			sess.run(tf.global_variables_initializer())
			sess.graph.finalize()
			sess.run(self.copy_weight)		# copy weights
			discount = [self.args.gamma] * self.args.batch_size
			for i_epoch in range(self.args.epoch):
				for j in range(num_batches):
					state, len_state, next_state, len_next_states, target_items, is_done = self.sample_data()
					actions = sess.run(self.main_agent.actor_out_, feed_dict={
						self.main_agent.inputs: state, 
						self.main_agent.len_state: len_state,
						self.main_agent.is_training: False})
					
					# add noise
					noise = np.random.normal(0, self.args.noise_var, size=self.main_agent.action_size).clip(-self.args.noise_clip, self.args.noise_clip)
					actions = (actions + noise).clip(-self.args.max_action, self.args.max_action)

					ce_loss, ranking_model_loss, _ = sess.run([
						self.main_agent.ce_loss,
						self.main_agent.ranking_model_loss, 
						self.main_agent.model_optim], 
						feed_dict={
						self.main_agent.inputs: state, 
						self.main_agent.len_state: len_state,
						self.main_agent.actor_out_: actions,
						# self.main_agent.actions: actions,
						self.main_agent.target_items: target_items,
						self.main_agent.is_training: True})

					if self.args.reward == 'ndcg':
						# target logits
						logits = sess.run(self.target_agent.logits,
							feed_dict={
							self.target_agent.inputs: state, 
							self.target_agent.len_state: len_state,
							self.target_agent.actor_out_: actions,
							# self.target_agent.actions: actions,
							self.target_agent.is_training: False})
						rewards = self.cal_rewards(logits, target_items)
						# true_next_state, true_next_state_len = self.state_trans(rewards, state, next_state, len_state, len_next_states)
					elif self.args.reward == 'loss':
						# ce_loss = sess.run(self.target_agent.ce_loss, feed_dict={
						# 	self.target_agent.inputs: state, 
						# 	self.target_agent.len_state: len_state,
						# 	self.target_agent.actor_out_: actions,
						# 	# self.target_agent.actions: actions,
						# 	self.target_agent.target_items: target_items,
						# 	self.target_agent.is_training: False})
						rewards = loss_reward(ce_loss)

					target_v = sess.run(self.target_agent.critic_output, feed_dict={
						self.target_agent.inputs: next_state,
						self.target_agent.len_state: len_next_states,
						# self.target_agent.inputs: true_next_state,
						# self.target_agent.len_state: true_next_state_len,

						self.target_agent.is_training: False})
					target_v = target_v.squeeze()
					for index in range(self.args.batch_size):
						if is_done[index]:
							target_v[index] = 0.0

					critic_loss, _ = sess.run([self.main_agent.critic_loss, 
						self.main_agent.critic_optim], 
						feed_dict={self.main_agent.inputs:state, 
						self.main_agent.len_state:len_state,
						self.main_agent.actor_out_: actions, 
						self.main_agent.reward: rewards,
						self.main_agent.discount: discount,
						self.main_agent.target: target_v,
						self.main_agent.is_training: True})
					actor_loss, _ = sess.run([self.main_agent.actor_loss, self.main_agent.actor_optim],
						feed_dict={self.main_agent.inputs: state, 
						self.main_agent.len_state: len_state,
						self.main_agent.is_training: True})
					sess.run(self.target_network_update_ops)		# update target net

					total_step += 1
					if (total_step == 1) or (total_step % 200 == 0):
						aver_reward = round(np.array(rewards).mean().item(), 5)
						ranking_model_loss, actor_loss, critic_loss = round(ranking_model_loss.item(), 5), round(actor_loss.item(), 5), round(critic_loss.item(), 5)
						info = f"epoch:{i_epoch} Step:{total_step}, aver reward:{aver_reward}, ranking model loss:{ranking_model_loss}, actor loss:{actor_loss}, critic loss:{critic_loss}"
						print(info)
						logging.info(info)
					if (total_step >= self.args.start_eval) and (total_step % self.args.eval_interval == 0):
						t1 = time.time()
						evaluate_multi_head(self.args, self.main_agent, sess, max_ndcg_and_epoch, total_step, logging)
						t2 = time.time()
						print(f'Time:{t2 - t1}')
						logging.info(f'Time:{t2 - t1}')


def main(args):
	tf.reset_default_graph()
	main_agent = Agent(args, name='train', max_action=args.max_action)
	target_agent = Agent(args, name='target', max_action=args.max_action)

	run = Run(args, main_agent, target_agent)
	run.train()


def parse_args():
	base_data_dir = '../../../data/'
	parser = argparse.ArgumentParser(description="SASRec DDPG.")
	parser.add_argument('--v', default="v")
	parser.add_argument('--mode', default='valid')
	parser.add_argument('--seed', type=int, default=-1)
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_data_dir', default=base_data_dir + 'RC15')
	parser.add_argument('--topk', default='5,10,20')

	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--eval_interval', type=int, default=2000)
	parser.add_argument('--start_eval', type=int, default=2000)
	parser.add_argument('--eval_batch', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--mlr', type=float, default=1e-3)
	parser.add_argument('--alr', type=float, default=1e-3)
	parser.add_argument('--clr', type=float, default=3e-4)

	parser.add_argument('--reward_top', type=int, default=20)

	parser.add_argument('--max_iid', type=int, default=26701)	# 0~26702
	parser.add_argument('--hidden_factor', type=int, default=64)

	parser.add_argument('--num_heads', default=1, type=int)
	parser.add_argument('--num_blocks', default=1, type=int)

	parser.add_argument('--dropout_rate', default=0.1, type=float)
	parser.add_argument('--weight_decay', default=1e-4, type=float)

	parser.add_argument('--noise_var', type=float, default=0.01)
	parser.add_argument('--noise_clip', type=float, default=0.05)
	parser.add_argument('--tau', type=float, default=0.001)
	parser.add_argument('--gamma', type=float, default=0.5)

	parser.add_argument('--note', default='None...')
	parser.add_argument('--mem_ratio', type=float, default=0.2)
	parser.add_argument('--cuda', default='0')
	parser.add_argument('--reward', default='ndcg')
	parser.add_argument('--max_action', type=float, default=0.1)
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

if __name__ == '__main__':
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	if args.seed != -1:
		random.seed(args.seed)
		np.random.seed(args.seed)
		tf.set_random_seed(args.seed)
	init_log(args)
	main(args)