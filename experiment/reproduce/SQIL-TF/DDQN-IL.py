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
			self.demo_reward = tf.placeholder(tf.float32, [None], name='demo_reward')
			self.samp_reward = tf.placeholder(tf.float32, [None], name='samp_reward')

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
				activation_fn=None, 
				weights_regularizer=tf.contrib.layers.l2_regularizer(args.weight_decay))

			self.demo_actions = tf.placeholder(tf.int32, [None], name='demo_actions')
			self.samp_actions = tf.placeholder(tf.int32, [None], name='samp_actions')
			self.samp_output = tf.placeholder(tf.float32, [None, self.item_num], name='samp_output')

			self.demo_targetQ = tf.placeholder(tf.float32, [None, self.item_num], name='demo_target')
			self.demo_targetQ_selector = tf.placeholder(tf.float32, [None, self.item_num], 
				name='demo_targetQ_selector')

			self.samp_targetQ = tf.placeholder(tf.float32, [None, self.item_num], name='samp_target')
			self.samp_targetQ_selector = tf.placeholder(tf.float32, [None, self.item_num], 
				name='samp_targetQ_selector')

			demo_loss, q_learning = trfl.double_qlearning(self.output, self.demo_actions, 
				self.demo_reward, self.discount, self.demo_targetQ, self.demo_targetQ_selector)
			samp_loss, q_learning = trfl.double_qlearning(self.samp_output, self.samp_actions, 
				self.samp_reward, self.discount, self.samp_targetQ, self.samp_targetQ_selector)

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
		rec_list = sorted_list[:, :topk[i]]
		for j in range(len(true_items)):
			for rank in range(len(rec_list[j])):
				if rec_list[j][rank].item() == true_items[j]:
					total_reward[i] += rewards[j]
					if rewards[j] == r_click:
						hit_click[i] += 1.0
						ndcg_click[i] += 1.0 / np.log2(rank + 2.0).item()
					else:
						hit_purchase[i] += 1.0
						ndcg_purchase[i] += 1.0 / np.log2(rank + 2.0).item()


def evaluate(args, trainQ, sess, max_ndcg_and_epoch, total_step):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	topk = [int(x) for x in args.topk.split(',')]
	max_topk = max(topk)
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
		prediction = torch.tensor(prediction)
		_, sorted_list = prediction.topk(max_topk)
		del prediction
		calculate_hit(sorted_list, topk, actions, rewards, args.reward_click, total_reward, hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)
	info = f'total clicks:{total_clicks}, total purchase:{total_purchase}'
	print(info)
	logging.info(info)
	total_reward = [round(x, 5) for x in total_reward]
	hr_click_list, hr_purchase_list, ng_click_list, ng_purchase_list = [], [], [], []
	for i in range(len(topk)):
		hr_click = hit_clicks[i] / total_clicks
		hr_purchase = hit_purchase[i] / total_purchase
		ng_click = ndcg_clicks[i] / total_clicks
		ng_purchase = ndcg_purchase[i] / total_purchase

		hr_click, hr_purchase = round(hr_click, 6), round(hr_purchase, 6)
		ng_click, ng_purchase = round(ng_click, 6), round(ng_purchase, 6)

		tup = max_ndcg_and_epoch[i]		# (ng_click, ng_purchase, step)
		if ng_click > tup[0]:
			max_ndcg_and_epoch[i][0] = ng_click
			max_ndcg_and_epoch[i][1] = ng_purchase
			max_ndcg_and_epoch[i][2] = total_step

		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		info = f'cumulative reward @ {topk[i]}: {total_reward[i]}'
		print(info)
		logging.info(info)
		info = f'clicks @ {topk[i]} : HR: {hr_click}, NDCG: {ng_click}'
		print(info)
		logging.info(info)
		info = f'purchase @ {topk[i]} : HR: {hr_purchase}, NDCG: {ng_purchase}'
		print(info)
		logging.info(info)
		info = f'Current Max click NDCG:{max_ndcg_and_epoch[i][0]}, the purchase NDCG is {max_ndcg_and_epoch[i][1]}. (step:{max_ndcg_and_epoch[i][2]})'
		print(info)
		logging.info(info)


def get_samp_data(sess, batch, trainQ):
	samp_state = list(batch['state'].values())
	samp_len_state = list(batch['len_state'].values())
	samp_tartget_actions = list(batch['action'].values())
	samp_target_next_state = list(batch['next_state'].values())
	samp_target_next_state_len = list(batch['len_next_states'].values())
	samp_is_done = list(batch['is_done'].values())

	samp_q = sess.run(trainQ.output, feed_dict={trainQ.inputs: samp_state, trainQ.len_state:samp_len_state})
	samp_actions = np.argmax(samp_q, 1)

	samp_next_states = []
	samp_next_states_len = []
	for i in range(len(samp_tartget_actions)):
		action, demo_action = samp_actions[i], samp_tartget_actions[i]
		samp_next_states.append(samp_target_next_state[i] if action == demo_action else samp_state[i])
		samp_next_states_len.append(samp_target_next_state_len[i] if action == demo_action else samp_len_state[i])
	return samp_q, samp_next_states, samp_next_states_len, samp_actions, samp_is_done


def main(args):
	tf.reset_default_graph()
	trainQ = DQN(args, name='trainQ')
	targetQ = DQN(args, name='targetQ')
	target_network_update_ops = trfl.update_target_variables(targetQ.get_qnetwork_variables(), 
		trainQ.get_qnetwork_variables(), tau=args.tau)

	replay_buffer = pd.read_pickle(os.path.join(args.base_data_dir, 'replay_buffer.df'))
	num_rows = replay_buffer.shape[0]
	num_batches = int(num_rows / args.batch_size)
	max_ndcg_and_epoch = [[0, 0, 0] for _ in args.topk.split(',')]	# (ng_click, ng_purchase, step)
	total_step = 0
	loss_list = []

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.graph.finalize()
		discount = [args.gamma] * args.batch_size
		demo_reward = [1.0] * args.batch_size
		samp_reward = [0.0] * args.batch_size
		for i_epoch in range(args.epoch):
			for j in range(num_batches):
				batch = replay_buffer.sample(n=args.batch_size).to_dict()
				state = list(batch['state'].values())
				len_state = list(batch['len_state'].values())
				next_state = list(batch['next_state'].values())
				len_next_states = list(batch['len_next_states'].values())
				demo_actions = list(batch['action'].values())
				is_done = list(batch['is_done'].values())

				# sample samp_data
				batch = replay_buffer.sample(n=args.batch_size).to_dict()
				samp_q, samp_next_states, samp_next_states_len, samp_actions, samp_is_done = get_samp_data(sess, batch, trainQ)

				demo_target = sess.run(targetQ.output, feed_dict={targetQ.inputs: next_state,
					targetQ.len_state:len_next_states})
				samp_target = sess.run(targetQ.output, feed_dict={targetQ.inputs: samp_next_states,
					targetQ.len_state:samp_next_states_len})

				# get selector
				demo_targetQ_selector = sess.run(trainQ.output, feed_dict={trainQ.inputs: next_state,
					trainQ.len_state:len_next_states})
				samp_targetQ_selector = sess.run(trainQ.output, feed_dict={trainQ.inputs: samp_next_states,
					trainQ.len_state:samp_next_states_len})

				for index in range(args.batch_size):
					if is_done[index]:
						demo_target[index] = np.zeros([trainQ.item_num])
					if samp_is_done[index]:
						samp_target[index] = np.zeros([trainQ.item_num])

				loss, _ = sess.run([trainQ.loss, trainQ.optim],
								   feed_dict={trainQ.inputs: state,
											  trainQ.len_state: len_state,
											  trainQ.samp_output: samp_q,
											  trainQ.demo_actions: demo_actions,
											  trainQ.samp_actions: samp_actions,
											  trainQ.demo_targetQ: demo_target,
											  trainQ.samp_targetQ: samp_target,
											  trainQ.demo_targetQ_selector: demo_targetQ_selector,
											  trainQ.samp_targetQ_selector: samp_targetQ_selector,
											  trainQ.discount: discount,
											  trainQ.demo_reward: demo_reward,
											  trainQ.samp_reward: samp_reward})
				sess.run(target_network_update_ops)		# update target net
				total_step += 1
				if (total_step == 1) or (total_step % 200 == 0):
					loss = round(loss.item(), 5)
					info = f"epoch:{i_epoch} Step:{total_step}, loss:{loss}"
					print(info)
					logging.info(info)
					loss_list.append(loss)
				if total_step % args.eval_interval == 0:
					t1 = time.time()
					evaluate(args, trainQ, sess, max_ndcg_and_epoch, total_step)
					t2 = time.time()
					print(f'Time:{t2 - t1}')
					logging.info(f'Time:{t2 - t1}')

	plt.figure(figsize=(8, 8))
	plt.title('Loss')
	plt.xlabel('Step')
	plt.ylabel('Precision')
	plt.plot(loss_list)
	plt.savefig(args.base_pic_dir + args.v + '.png')


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
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--eval_interval', type=int, default=6000)
	parser.add_argument('--eval_batch', type=int, default=10)

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
	parser.add_argument('--lambda_samp', type=float, default=1.0)

	parser.add_argument('--seq_hidden_size', type=int, default=64)
	parser.add_argument('--seq_layer_num', type=int, default=1)

	parser.add_argument('--tau', type=float, default=0.01)
	parser.add_argument('--gamma', type=float, default=0.6)
	parser.add_argument('--layer_trick', default='none')			# ln/bn/none
	parser.add_argument('--dropout', type=float, default=0.0)
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	init_log(args)
	main(args)