import os
from collections import deque

import tensorflow as tf
import torch
import pandas as pd
import numpy as np

def count2():
    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    assert 0>1

def loss_reward(ce_loss):
	rewards = -ce_loss.reshape((-1))
	return rewards

def hit_reward(args, logits, target_items):
	logits = torch.tensor(logits)
	_, rankings = logits.topk(args.reward_top)
	rankings = rankings.tolist()	# (batch, topk)
	rewards = []
	for target_iid, rec_list in zip(target_items, rankings):
		rewards.append(1.0 if target_iid in set(rec_list) else args.init_r) 	# init_r: don't hit reward (0/-1)
	return rewards


def mlp(x, is_training, hidden_sizes=(32,), activation=tf.nn.relu, output_activation=tf.nn.tanh, 
	dropout_rate=0.1, l2=None):
	for h in hidden_sizes[:-1]:
		x = tf.layers.dense(x, units=h, activation=activation, activity_regularizer=l2)
		x = tf.layers.dropout(x, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
	return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, activity_regularizer=l2)


def extract_axis_1(data, ind):
	"""
	Get specified elements along the first axis of tensor.
	:param data: Tensorflow tensor that will be subsetted.
	:param ind: Indices to take (one for each element along axis 0 of data).
	:return: Subsetted tensor.
	"""

	batch_range = tf.range(tf.shape(data)[0])
	indices = tf.stack([batch_range, ind], axis=1)
	res = tf.gather_nd(data, indices)

	return res

def normalize(inputs,
			  epsilon=1e-8,
			  scope="ln",
			  reuse=None):
	'''Applies layer normalization.

	Args:
	  inputs: A tensor with 2 or more dimensions, where the first dimension has
		`batch_size`.
	  epsilon: A floating number. A very small number for preventing ZeroDivision Error.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.

	Returns:
	  A tensor with the same shape and data dtype as `inputs`.
	'''
	with tf.variable_scope(scope, reuse=reuse):
		inputs_shape = inputs.get_shape()
		params_shape = inputs_shape[-1:]

		mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
		beta = tf.Variable(tf.zeros(params_shape))
		gamma = tf.Variable(tf.ones(params_shape))
		normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
		outputs = gamma * normalized + beta

	return outputs

def pad_history(itemlist, length, pad_item):
	if len(itemlist) >= length:
		return itemlist[-length:]
	if len(itemlist) < length:
		temp = [pad_item] * (length - len(itemlist))
		itemlist.extend(temp)
		return itemlist


def calculate_hit(sorted_list, topk, true_items, hit_inters, ndcg_inters):
	for i in range(len(topk)):
		rec_list = sorted_list[:, :topk[i]]
		for j in range(len(true_items)):
			for rank in range(len(rec_list[j])):
				if rec_list[j][rank].item() == true_items[j]:
					hit_inters[i] += 1.0
					ndcg_inters[i] += 1.0 / np.log2(rank + 2.0).item()


def print_eval(total_inter, hit_inters, ndcg_inters, topk, max_ndcg_and_epoch, total_step, logging, session_num):
	info = f'total interaction:{total_inter}'
	print(info)
	logging.info(info)
	for i in range(len(topk)):
		hr_inter = hit_inters[i] / total_inter
		ng_inter = ndcg_inters[i] / total_inter
		session_cumulative_reward = hit_inters[i] / session_num

		hr_inter, ng_inter, session_cumulative_reward = round(hr_inter, 4), round(ng_inter, 4), round(session_cumulative_reward, 4)

		tup = max_ndcg_and_epoch[i]		# (ng_inter, step)
		if ng_inter > tup[0]:
			max_ndcg_and_epoch[i][0] = ng_inter
			max_ndcg_and_epoch[i][1] = total_step

		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		info = f'@ {topk[i]} : HR: {hr_inter}, NDCG: {ng_inter}, Aver Session\'s cumulative reward: {session_cumulative_reward}'
		print(info)
		logging.info(info)
		info = f'Current Max NDCG:{max_ndcg_and_epoch[i][0]}. (step:{max_ndcg_and_epoch[i][1]})'
		print(info)
		logging.info(info)


def evaluate(args, ranking_model, sess, max_ndcg_and_epoch, total_step, logging, RL=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	topk = [int(x) for x in args.topk.split(',')]
	max_topk = max(topk)
	if args.mode == 'valid':
		eval_sessions = pd.read_pickle(os.path.join(args.base_data_dir, 'sampled_val.df'))
	elif args.mode == 'test':
		eval_sessions = pd.read_pickle(os.path.join(args.base_data_dir, 'sampled_test.df'))
	eval_ids = eval_sessions.session_id.unique()
	groups = eval_sessions.groupby('session_id')
	session_num = len(eval_ids)

	batch = args.eval_batch
	evaluated = 0
	total_inter = 0.0
	hit_inters, ndcg_inters = [0 for _ in topk], [0 for _ in topk]
	
	while evaluated < len(eval_ids):
		states, len_states, true_items = [], [], []
		for i in range(batch):
			if evaluated == len(eval_ids):
				break
			sid = eval_ids[evaluated]
			group = groups.get_group(sid)
			history = []
			for index, row in group.iterrows():
				state = list(history)
				len_states.append(ranking_model.hw if len(state) >= ranking_model.hw else 1 if len(state) == 0 else len(state))
				state = pad_history(state, ranking_model.hw, ranking_model.item_num)
				states.append(state)
				target_item = row['item_id']
				total_inter += 1.0
				true_items.append(target_item)
				history.append(row['item_id'])
			evaluated += 1
		if RL:
			prediction = sess.run(ranking_model.output2, feed_dict={ranking_model.inputs: states,
				ranking_model.len_state:len_states,
				ranking_model.is_training:False})
		else:
			prediction = sess.run(ranking_model.output, feed_dict={ranking_model.inputs: states, 
				ranking_model.len_state: len_states, 
				ranking_model.is_training: False})

		prediction = torch.tensor(prediction)
		_, sorted_list = prediction.topk(max_topk)
		del prediction
		calculate_hit(sorted_list, topk, true_items, hit_inters, ndcg_inters)
	print_eval(total_inter, hit_inters, ndcg_inters, topk, max_ndcg_and_epoch, total_step, logging, session_num)


def evaluate_multi_head(args, agent, sess, max_ndcg_and_epoch, total_step, logging):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	topk = [int(x) for x in args.topk.split(',')]
	max_topk = max(topk)
	if args.mode == 'valid':
		eval_sessions = pd.read_pickle(os.path.join(args.base_data_dir, 'sampled_val.df'))
	elif args.mode == 'test':
		eval_sessions = pd.read_pickle(os.path.join(args.base_data_dir, 'sampled_test.df'))
	eval_ids = eval_sessions.session_id.unique()
	groups = eval_sessions.groupby('session_id')
	session_num = len(eval_ids)

	batch = args.eval_batch
	evaluated = 0
	total_inter = 0.0
	hit_inters, ndcg_inters = [0 for _ in topk], [0 for _ in topk]
	
	while evaluated < len(eval_ids):
		states, len_states, true_items = [], [], []
		for i in range(batch):
			if evaluated == len(eval_ids):
				break
			sid = eval_ids[evaluated]
			group = groups.get_group(sid)
			history = []
			for index, row in group.iterrows():
				state = list(history)
				len_states.append(agent.hw if len(state) >= agent.hw else 1 if len(state) == 0 else len(state))
				state = pad_history(state, agent.hw, agent.item_num)
				states.append(state)
				target_item = row['item_id']
				total_inter += 1.0
				true_items.append(target_item)
				history.append(row['item_id'])
			evaluated += 1

		# actions = sess.run(agent.actor_out_, feed_dict={agent.inputs: states, 
		# 	agent.len_state: len_states,
		# 	agent.is_training: False})
		prediction = sess.run(agent.logits, feed_dict={
						agent.inputs: states, 
						agent.len_state: len_states,
						# agent.actor_out_: actions,
						agent.is_training: False})

		prediction = torch.tensor(prediction)
		_, sorted_list = prediction.topk(max_topk)
		del prediction
		calculate_hit(sorted_list, topk, true_items, hit_inters, ndcg_inters)
	print_eval(total_inter, hit_inters, ndcg_inters, topk, max_ndcg_and_epoch, total_step, logging, session_num)


def evaluate_with_actions(args, agent, sess, max_ndcg_and_epoch, total_step, logging):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	topk = [int(x) for x in args.topk.split(',')]
	max_topk = max(topk)
	if args.mode == 'valid':
		eval_sessions = pd.read_pickle(os.path.join(args.base_data_dir, 'sampled_val.df'))
	elif args.mode == 'test':
		eval_sessions = pd.read_pickle(os.path.join(args.base_data_dir, 'sampled_test.df'))
	eval_ids = eval_sessions.session_id.unique()
	groups = eval_sessions.groupby('session_id')
	session_num = len(eval_ids)

	batch = args.eval_batch
	evaluated = 0
	total_inter = 0.0
	hit_inters, ndcg_inters = [0 for _ in topk], [0 for _ in topk]
	
	while evaluated < len(eval_ids):
		states, len_states, true_items = [], [], []
		for i in range(batch):
			if evaluated == len(eval_ids):
				break
			sid = eval_ids[evaluated]
			group = groups.get_group(sid)
			history = []
			for index, row in group.iterrows():
				state = list(history)
				len_states.append(agent.hw if len(state) >= agent.hw else 1 if len(state) == 0 else len(state))
				state = pad_history(state, agent.hw, agent.item_num)
				states.append(state)
				target_item = row['item_id']
				total_inter += 1.0
				true_items.append(target_item)
				history.append(row['item_id'])
			evaluated += 1

		actions = sess.run(agent.actor_out_, feed_dict={agent.inputs: states, 
			agent.len_state: len_states,
			agent.is_training: False})
		prediction = sess.run(agent.logits, 
						feed_dict={
						agent.inputs: states, 
						agent.len_state: len_states,
						# agent.actor_out_: actions,
						agent.actions: actions,
						agent.is_training: False})

		prediction = torch.tensor(prediction)
		_, sorted_list = prediction.topk(max_topk)
		del prediction
		calculate_hit(sorted_list, topk, true_items, hit_inters, ndcg_inters)
	print_eval(total_inter, hit_inters, ndcg_inters, topk, max_ndcg_and_epoch, total_step, logging, session_num)

def evaluate_DDPG(args, ranking_model, sess, max_ndcg_and_epoch, total_step, logging):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	topk = [int(x) for x in args.topk.split(',')]
	max_topk = max(topk)
	if args.mode == 'valid':
		eval_sessions = pd.read_pickle(os.path.join(args.base_data_dir, 'sampled_val.df'))
	elif args.mode == 'test':
		eval_sessions = pd.read_pickle(os.path.join(args.base_data_dir, 'sampled_test.df'))
	eval_ids = eval_sessions.session_id.unique()
	groups = eval_sessions.groupby('session_id')
	session_num = len(eval_ids)

	batch = args.eval_batch
	evaluated = 0
	total_inter = 0.0
	hit_inters, ndcg_inters = [0 for _ in topk], [0 for _ in topk]
	
	while evaluated < len(eval_ids):
		states, len_states, true_items = [], [], []
		for i in range(batch):
			if evaluated == len(eval_ids):
				break
			sid = eval_ids[evaluated]
			group = groups.get_group(sid)
			history = []
			for index, row in group.iterrows():
				state = list(history)
				len_states.append(ranking_model.hw if len(state) >= ranking_model.hw else 1 if len(state) == 0 else len(state))
				state = pad_history(state, ranking_model.hw, ranking_model.item_num)
				states.append(state)
				target_item = row['item_id']
				total_inter += 1.0
				true_items.append(target_item)
				history.append(row['item_id'])
			evaluated += 1

		prediction = sess.run(ranking_model.scores, feed_dict={ranking_model.inputs: states, 
			ranking_model.len_state: len_states, 
			ranking_model.is_training: False})

		prediction = torch.tensor(prediction)
		_, sorted_list = prediction.topk(max_topk)
		del prediction
		calculate_hit(sorted_list, topk, true_items, hit_inters, ndcg_inters)
	print_eval(total_inter, hit_inters, ndcg_inters, topk, max_ndcg_and_epoch, total_step, logging, session_num)