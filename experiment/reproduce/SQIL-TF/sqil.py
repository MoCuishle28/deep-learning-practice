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


class SoftQ(object):
	def __init__(self, args, name='SoftQ'):
		super(SoftQ, self).__init__()
		self.args = args
		self.item_num = args.max_iid + 1
		self.hw = 10		# history window(过去多少次交互记录作为输入)
		self.is_training = tf.placeholder(tf.bool, shape=())
		self.name = name
		with tf.variable_scope(self.name):
			self.discount = tf.constant(args.gamma, shape=[args.batch_size], dtype=tf.float32, name="discount")
			self.demo_reward = tf.constant(1.0, dtype=tf.float32, shape=[args.batch_size], name='demo_reward')
			self.samp_reward = tf.constant(0.0, dtype=tf.float32, shape=[args.batch_size], name='samp_reward')

			self.inputs = tf.placeholder(tf.int32, [None, self.hw], name='inputs')
			self.len_state = tf.placeholder(tf.int32, [None], name='len_state')

			self.all_embeddings = self.initialize_embeddings()
			self.item_emb = tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'], self.inputs)

			gru_out, self.states_hidden = tf.nn.dynamic_rnn(
				tf.contrib.rnn.GRUCell(self.args.seq_hidden_size), self.item_emb,
				dtype = tf.float32,
				sequence_length = self.len_state,
			)
			# self.output = tf.layers.Dense(self.states_hidden, self.item_num)
			self.output = tf.contrib.layers.fully_connected(self.states_hidden, self.item_num, 
				activation_fn=None)

			self.demo_actions = tf.placeholder(tf.int32, [None], name='demo_actions')
			self.samp_actions = tf.placeholder(tf.int32, [None], name='samp_actions')

			self.demo_targetQ = tf.placeholder(tf.float32, [None, self.item_num], name='demo_target')
			self.samp_targetQ = tf.placeholder(tf.float32, [None, self.item_num], name='samp_target')

			# target
			demo_target = tf.log(tf.reduce_sum(tf.exp(self.demo_targetQ), 1))		# (batch)
			samp_target = tf.log(tf.reduce_sum(tf.exp(self.samp_targetQ), 1))		# (batch)

			demo_current_q = indexing_ops.batched_index(self.output, self.demo_actions)
			samp_current_q = indexing_ops.batched_index(self.output, self.samp_actions)

			'''
			TRFL td_learning: (v_tm1 [B], r_t [B], pcon_t [B], v_t [B])
			The TD loss is 0.5 times the squared difference between v_tm1 and the target r_t + pcont_t * v_t.
			'''
			demo_loss, q_learning = trfl.td_learning(demo_current_q, self.demo_reward, 
				self.discount, demo_target)
			samp_loss, q_learning = trfl.td_learning(samp_current_q, self.samp_reward, 
				self.discount, samp_target)

			self.demo_loss = tf.reduce_mean(demo_loss)
			self.samp_loss = tf.reduce_mean(samp_loss)
			self.loss = self.demo_loss + args.lambda_samp*self.samp_loss
			self.optim = self.get_optim(args.optim)


	def initialize_embeddings(self):
		all_embeddings = dict()
		item_embeddings = tf.Variable(tf.random_normal([self.args.max_iid + 2, self.args.i_emb_dim], 
			0.0, 0.01), name='item_embeddings')
		all_embeddings['item_embeddings'] = item_embeddings
		return all_embeddings


	def get_qnetwork_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


	def get_optim(self, key):
		optim = None
		if key == 'rms':
			optim = tf.train.RMSPropOptimizer(self.args.lr, momentum=self.args.momentum).minimize(self.loss)
		elif key == 'sgd':
			optim = tf.train.MomentumOptimizer(self.args.lr, momentum=self.args.momentum).minimize(self.loss)
		else:
			optim = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)
		return optim


def pad_history(itemlist, length, pad_item):
	if len(itemlist) >= length:
		return itemlist[-length:]
	if len(itemlist) < length:
		temp = [pad_item] * (length - len(itemlist))
		itemlist.extend(temp)
		return itemlist


def calculate_hit(sorted_list, topk, true_items, rewards, r_click, total_reward, hit_click, ndcg_click, hit_purchase, ndcg_purchase):
	for i in range(len(topk)):
		rec_list = sorted_list[:, -topk[i]:]
		for j in range(len(true_items)):
			if true_items[j] in rec_list[j]:
				rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
				total_reward[i] += rewards[j]
				if rewards[j] == r_click:
					hit_click[i] += 1.0
					ndcg_click[i] += 1.0 / np.log2(rank + 1)
				else:
					hit_purchase[i] += 1.0
					ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def evaluate(args, trainQ, sess):
	topk = [int(x) for x in args.topk.split(',')]
	if args.mode == 'valid':
		eval_sessions = pd.read_pickle(os.path.join(args.base_data_dir, 'sampled_val.df'))
	elif args.mode == 'test':
		eval_sessions = pd.read_pickle(os.path.join(args.base_data_dir, 'sampled_test.df'))
	eval_ids = eval_sessions.session_id.unique()
	groups = eval_sessions.groupby('session_id')

	batch = args.eval_batch
	evaluated = 0
	total_clicks, total_purchase = 0.0, 0.0
	total_reward = [0 for _ in topk]
	hit_clicks, ndcg_clicks = [0 for _ in topk], [0 for _ in topk]
	hit_purchase, ndcg_purchase = [0 for _ in topk], [0 for _ in topk]
	
	while evaluated < len(eval_ids):
		states, len_states, actions, rewards = [], [], [], []
		for i in range(batch):
			if evaluated == len(eval_ids):
				break
			sid = eval_ids[evaluated]
			group = groups.get_group(sid)
			history = []
			for index, row in group.iterrows():
				state = list(history)
				len_states.append(trainQ.hw if len(state) >= trainQ.hw else 1 if len(state) == 0 else len(state))
				state = pad_history(state, trainQ.hw, trainQ.item_num)
				states.append(state)
				action = row['item_id']
				is_buy = row['is_buy']
				reward = args.reward_buy if is_buy == 1 else args.reward_click
				if is_buy == 1:
					total_purchase += 1.0
				else:
					total_clicks += 1.0
				actions.append(action)
				rewards.append(reward)
				history.append(row['item_id'])
			evaluated += 1
		prediction = sess.run(trainQ.output, feed_dict={trainQ.inputs: states, trainQ.len_state: len_states})
		sorted_list = np.argsort(prediction)	# 返回从小到大的索引
		calculate_hit(sorted_list, topk, actions, rewards, args.reward_click, total_reward, hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)
	info = f'total clicks:{total_clicks}, total purchase:{total_purchase}'
	print(info)
	logging.info(info)
	total_reward = [round(x, 5) for x in total_reward]
	for i in range(len(topk)):
		hr_click = hit_clicks[i] / total_clicks
		hr_purchase = hit_purchase[i] / total_purchase
		ng_click = ndcg_clicks[i] / total_clicks
		ng_purchase = ndcg_purchase[i] / total_purchase

		hr_click, hr_purchase = round(hr_click, 5), round(hr_purchase)
		ng_click, ng_purchase = round(ng_click, 5), round(ng_purchase, 5)
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		loggin.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		info = f'cumulative reward @ {topk[i]}: {total_reward[i]}'
		print(info)
		logging.info(info)
		info = f'clicks hr ndcg @ {topk[i]} : {hr_click}, {ng_click}'
		print(info)
		logging.info(info)
		info = f'purchase hr and ndcg @{topk[i]} : {hr_purchase}, {ng_purchase}'
		print(info)
		logging.info(info)


def main(args):
	trainQ = SoftQ(args, name='trainQ')
	if args.target == 'y':
		targetQ = SoftQ(args, name='targetQ')
		target_network_update_ops = trfl.update_target_variables(targetQ.get_qnetwork_variables(), 
			trainQ.get_qnetwork_variables(), tau=args.tau)

	replay_buffer = pd.read_pickle(os.path.join(args.base_data_dir, 'replay_buffer.df'))
	num_rows = replay_buffer.shape[0]
	num_batches = int(num_rows / args.batch_size)
	total_step = 0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(args.epoch):
			for j in range(num_batches):
				batch = replay_buffer.sample(n=args.batch_size).to_dict()
				state = list(batch['state'].values())
				len_state = list(batch['len_state'].values())
				next_state = list(batch['next_state'].values())
				len_next_states = list(batch['len_next_states'].values())
				demo_actions = list(batch['action'].values())

				samp_q = sess.run(trainQ.output, feed_dict={trainQ.inputs: state, 
					trainQ.len_state:len_state})
				samp_actions = tf.argmax(samp_q, 1).eval()

				samp_next_states = []
				samp_next_states_len = []
				for i in range(len(demo_actions)):
					action, demo_action = samp_actions[i], demo_actions[i]
					samp_next_states.append(next_state[i] if action == demo_action else state[i])
					samp_next_states_len.append(len_next_states[i] if action == demo_action else len_state[i])

				if args.target == 'y':
					demo_target = sess.run(targetQ.output, feed_dict={targetQ.inputs: next_state,
						targetQ.len_state:len_next_states})
					samp_target = sess.run(targetQ.output, feed_dict={targetQ.inputs: samp_next_states,
						targetQ.len_state:samp_next_states_len})
				else:
					demo_target = sess.run(trainQ.output, feed_dict={trainQ.inputs: next_state,
						trainQ.len_state:len_next_states})
					samp_target = sess.run(trainQ.output, feed_dict={trainQ.inputs: samp_next_states,
						trainQ.len_state:samp_next_states_len})

				is_done = list(batch['is_done'].values())
				for index in range(demo_target.shape[0]):
					if is_done[index]:
						demo_target[index] = np.zeros([trainQ.item_num])
						samp_target[index] = np.zeros([trainQ.item_num])

				loss, _ = sess.run([trainQ.loss, trainQ.optim],
								   feed_dict={trainQ.inputs: state,
											  trainQ.len_state: len_state,
											  trainQ.demo_actions: demo_actions,
											  trainQ.samp_actions: samp_actions,
											  trainQ.demo_targetQ: demo_target,
											  trainQ.samp_targetQ: samp_target})
				total_step += 1
				if total_step % 200 == 0:
					loss = round(loss.item(), 5)
					info = f"Step:{total_step}, loss:{loss}"
					print(info)
					logging.info(info)
				if total_step % args.eval_interval == 0:
					evaluate(args, trainQ, sess)


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
	parser.add_argument('--base_data_dir', default=base_data_dir + 'kaggle-RL4REC')
	parser.add_argument('--show', default='n')
	parser.add_argument('--mode', default='valid')		# test/valid
	parser.add_argument('--target', default='y')		# n/y -> target net
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--eval_interval', type=int, default=2000)
	parser.add_argument('--eval_batch', type=int, default=30)

	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--topk', default='5,10,20')
	parser.add_argument('--batch_size', type=int, default=256)

	parser.add_argument('--optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)
	parser.add_argument('--lr', type=float, default=1e-3)
	# embedding
	parser.add_argument('--max_iid', type=int, default=70851)	# 0~70851
	parser.add_argument('--i_emb_dim', type=int, default=64)
	# Soft Q
	parser.add_argument('--reward_buy', type=float, default=1.0)
	parser.add_argument('--reward_click', type=float, default=0.5)

	parser.add_argument('--seq_hidden_size', type=int, default=64)
	parser.add_argument('--seq_layer_num', type=int, default=1)

	parser.add_argument('--lambda_samp', type=float, default=1.0)
	parser.add_argument('--tau', type=float, default=0.1)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--lammbda_samp', type=float, default=1.0)
	parser.add_argument('--layer_trick', default='none')			# ln/bn/none
	parser.add_argument('--dropout', type=float, default=0.0)
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	init_log(args)
	main(args)