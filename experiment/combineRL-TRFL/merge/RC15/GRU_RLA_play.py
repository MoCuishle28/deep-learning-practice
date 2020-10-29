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
			self.reward = tf.placeholder(tf.float32, [None], name='reward')
			self.inputs = tf.placeholder(tf.int32, [None, self.hw], name='inputs')
			self.len_state = tf.placeholder(tf.int32, [None], name='len_state')
			self.target = tf.placeholder(tf.float32, [None], name='target')

			self.all_embeddings = self.initialize_embeddings()
			self.item_emb = tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'], self.inputs)

			gru_out, self.states_hidden = tf.nn.dynamic_rnn(
				tf.contrib.rnn.GRUCell(self.args.seq_hidden_size), self.item_emb,
				dtype = tf.float32,
				sequence_length = self.len_state,
			)
			if args.layer_trick == 'ln':
				self.states_hidden = tf.contrib.layers.layer_norm(self.states_hidden)

			self.actor_output = tf.contrib.layers.fully_connected(tf.stop_gradient(self.states_hidden), args.action_size, 
				activation_fn=tf.nn.tanh, 
				weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))
			self.actor_out_ = self.actor_output * max_action

			self.critic_input = tf.concat([self.actor_out_, tf.stop_gradient(self.states_hidden)], axis=1)
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

			# ranking model
			self.target_items = tf.placeholder(tf.int32, [None], name='target_items')

			self.ranking_model_input = self.actor_out_ + self.states_hidden

			self.logits = tf.contrib.layers.fully_connected(self.ranking_model_input, 
				args.max_iid + 1, 
				activation_fn=None, 
				weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))

			self.ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_items, 
				logits=self.logits)
			self.ranking_model_loss = tf.reduce_mean(self.ce_loss)
			self.model_optim = tf.train.AdamOptimizer(args.mlr).minimize(self.ranking_model_loss)

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
		self.main_agent = main_agent
		self.target_agent = target_agent
		self.target_network_update_ops = trfl.update_target_variables(target_agent.get_qnetwork_variables(), 
			main_agent.get_qnetwork_variables(), tau=args.tau)
		self.copy_weight = trfl.update_target_variables(target_agent.get_qnetwork_variables(), 
			main_agent.get_qnetwork_variables(), tau=1.0)

		self.topk = [int(x) for x in args.topk.split(',')]
		self.replay_buffer = pd.read_pickle(os.path.join(args.base_data_dir, 'replay_buffer.df'))
		self.memory = ReplayMemory(args.maxlen)

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
					ui_seq, ui_seq_len, _, _, target_items, _ = self.sample_data()
					state = sess.run(self.main_agent.states_hidden, feed_dict={
						self.main_agent.inputs: ui_seq,
						self.main_agent.len_state: ui_seq_len,
						self.main_agent.is_training: False
						})

					interaction_time = 0
					while True:
						actions = sess.run(self.main_agent.actor_out_, feed_dict={
							self.main_agent.states_hidden: state,
							self.main_agent.is_training: False})
						# add noise
						noise = np.random.normal(0, self.args.noise_var, size=self.args.action_size).clip(-self.args.noise_clip, self.args.noise_clip)
						actions = (actions + noise).clip(-self.args.max_action, self.args.max_action)

						if interaction_time == 0:
							ce_loss, ranking_model_loss, _ = sess.run([
								self.main_agent.ce_loss,
								self.main_agent.ranking_model_loss, 
								self.main_agent.model_optim], 
								feed_dict={
								self.main_agent.inputs: ui_seq, 
								self.main_agent.len_state: ui_seq_len,
								self.main_agent.actor_out_: actions,
								self.main_agent.target_items: target_items,
								self.main_agent.is_training: True})
						# get new state
						state_new = sess.run(self.main_agent.ranking_model_input, 
							feed_dict={
							self.main_agent.states_hidden: state, 
							self.main_agent.actor_out_: actions,
							self.main_agent.is_training: False})

						# target logits
						logits = sess.run(self.target_agent.logits,
							feed_dict={
							self.target_agent.states_hidden: state, 
							self.target_agent.actor_out_: actions,
							self.target_agent.is_training: False})
						rewards = self.cal_rewards(logits, target_items)	# get rewards

						state_new_list = []		# continue play
						for idx, (s, a, s_, r) in enumerate(zip(state, actions, state_new, rewards)):
							if r == 0: 		# continue play
								state_new_list.append(s_)
							done = True if r != 0 else False
							self.memory.push(s, a, s_, r, done)

						if interaction_time % self.args.train_interval == 0:
							transitions = self.memory.sample(self.args.batch_size)
							batch = Transition(*zip(*transitions))
							state, actions = batch.state, batch.action
							rewards, next_state, is_done = batch.reward, batch.next_state, batch.is_done

							target_v = sess.run(self.target_agent.critic_output, feed_dict={
								self.target_agent.states_hidden: next_state,
								self.target_agent.is_training: False})
							target_v = target_v.squeeze()
							for index in range(self.args.batch_size):
								if is_done[index]:
									target_v[index] = 0.0

							critic_loss, _ = sess.run([self.main_agent.critic_loss, 
								self.main_agent.critic_optim], 
								feed_dict={self.main_agent.states_hidden:state, 
								self.main_agent.actor_out_: actions, 
								self.main_agent.reward: rewards,
								self.main_agent.discount: discount,
								self.main_agent.target: target_v,
								self.main_agent.is_training: True})
							actor_loss, _ = sess.run([self.main_agent.actor_loss, 
								self.main_agent.actor_optim],
								feed_dict={self.main_agent.states_hidden: state, 
								self.main_agent.is_training: True})
							sess.run(self.target_network_update_ops)		# update target net
							info = f'Step:[{total_step}] train agent, batch size:{len(state_new_list)} V:{self.args.v}'
							print(info)
							# logging.info(info)
						interaction_time += 1
						state = state_new_list		# next state
						if (state_new_list == []) or (interaction_time >= self.args.max_interaction):
							print('Interaction END!')
							break

					total_step += 1
					if (total_step == 1) or (total_step % 200 == 0):
						ranking_model_loss, actor_loss, critic_loss = round(ranking_model_loss.item(), 5), round(actor_loss.item(), 5), round(critic_loss.item(), 5)
						info = f"epoch:{i_epoch} Step:{total_step}, ranking model loss:{ranking_model_loss}, actor loss:{actor_loss}, critic loss:{critic_loss}"
						print(info)
						logging.info(info)
					if (total_step >= self.args.start_eval) and (total_step % self.args.eval_interval == 0):
						t1 = time.time()
						evaluate_multi_head(self.args, self.main_agent, sess, max_ndcg_and_epoch, total_step, logging)
						# evaluate_with_actions(self.args, self.main_agent, sess, max_ndcg_and_epoch, total_step, logging)
						t2 = time.time()
						print(f'Time:{t2 - t1}')
						logging.info(f'Time:{t2 - t1}')


def main(args):
	tf.reset_default_graph()
	main_agent = Agent(args, name='train', max_action=args.max_action)
	target_agent = Agent(args, name='target', max_action=args.max_action)

	run = Run(args, main_agent, target_agent)
	run.train()


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
	base_data_dir = '../../../data/'
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--v', default="v")
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_data_dir', default=base_data_dir + 'RC15')
	parser.add_argument('--mode', default='valid')		# test/valid
	parser.add_argument('--seed', type=int, default=-1)
	parser.add_argument('--eval_interval', type=int, default=2000)
	parser.add_argument('--start_eval', type=int, default=2000)
	parser.add_argument('--eval_batch', type=int, default=10)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--topk', default='5,10,20')

	parser.add_argument('--weight_decay', type=float, default=1e-4)
	# embedding
	parser.add_argument('--max_iid', type=int, default=26701)	# 0~26702
	parser.add_argument('--i_emb_dim', type=int, default=64)

	parser.add_argument('--reward_top', type=int, default=20)		# 取 top 多少计算 reward

	parser.add_argument('--seq_hidden_size', type=int, default=64)
	parser.add_argument('--action_size', type=int, default=64)
	parser.add_argument('--mlr', type=float, default=1e-3)
	parser.add_argument('--alr', type=float, default=1e-3)
	parser.add_argument('--clr', type=float, default=3e-4)

	parser.add_argument('--noise_var', type=float, default=0.01)
	parser.add_argument('--noise_clip', type=float, default=0.05)
	parser.add_argument('--tau', type=float, default=0.001)
	parser.add_argument('--gamma', type=float, default=0.5)
	parser.add_argument('--layer_trick', default='ln')			# ln/bn/none

	parser.add_argument('--note', default='None...')
	parser.add_argument('--mem_ratio', type=float, default=0.2)
	parser.add_argument('--cuda', default='0')
	parser.add_argument('--reward', default='ndcg')
	parser.add_argument('--max_action', type=float, default=0.1)
	parser.add_argument('--maxlen', type=int, default=10000)
	parser.add_argument('--max_interaction', type=int, default=300)
	parser.add_argument('--train_interval', type=int, default=100)
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	if args.seed != -1:
		random.seed(args.seed)
		np.random.seed(args.seed)
		tf.set_random_seed(args.seed)

	init_log(args)
	main(args)