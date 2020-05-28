import os
from collections import deque

import tensorflow as tf
import torch
import pandas as pd
import numpy as np


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


def evaluate(args, ranking_model, sess, max_ndcg_and_epoch, total_step, logging):
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
		states, len_states, true_items, rewards = [], [], [], []
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
				is_buy = row['is_buy']
				reward = args.reward_buy if is_buy == 1 else args.reward_click
				if is_buy == 1:
					total_purchase += 1.0
				else:
					total_clicks += 1.0
				true_items.append(target_item)
				rewards.append(reward)
				history.append(row['item_id'])
			evaluated += 1

		prediction = sess.run(ranking_model.output, feed_dict={ranking_model.inputs: states,
											  ranking_model.is_training: False})
		prediction = torch.tensor(prediction)
		_, sorted_list = prediction.topk(max_topk)
		del prediction
		calculate_hit(sorted_list, topk, true_items, rewards, args.reward_click, total_reward, hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)
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

		hr_click, hr_purchase = round(hr_click, 5), round(hr_purchase, 5)
		ng_click, ng_purchase = round(ng_click, 5), round(ng_purchase, 5)

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

	batch = args.eval_batch
	evaluated = 0
	total_clicks, total_purchase = 0.0, 0.0
	total_reward = [0 for _ in topk]
	hit_clicks, ndcg_clicks = [0 for _ in topk], [0 for _ in topk]
	hit_purchase, ndcg_purchase = [0 for _ in topk], [0 for _ in topk]
	
	while evaluated < len(eval_ids):
		states, len_states, true_items, rewards = [], [], [], []
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
				is_buy = row['is_buy']
				reward = args.reward_buy if is_buy == 1 else args.reward_click
				if is_buy == 1:
					total_purchase += 1.0
				else:
					total_clicks += 1.0
				true_items.append(target_item)
				rewards.append(reward)
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
		calculate_hit(sorted_list, topk, true_items, rewards, args.reward_click, total_reward, hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)
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

		hr_click, hr_purchase = round(hr_click, 4), round(hr_purchase, 4)
		ng_click, ng_purchase = round(ng_click, 4), round(ng_purchase, 4)

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

	batch = args.eval_batch
	evaluated = 0
	total_clicks, total_purchase = 0.0, 0.0
	total_reward = [0 for _ in topk]
	hit_clicks, ndcg_clicks = [0 for _ in topk], [0 for _ in topk]
	hit_purchase, ndcg_purchase = [0 for _ in topk], [0 for _ in topk]
	
	while evaluated < len(eval_ids):
		states, len_states, true_items, rewards = [], [], [], []
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
				is_buy = row['is_buy']
				reward = args.reward_buy if is_buy == 1 else args.reward_click
				if is_buy == 1:
					total_purchase += 1.0
				else:
					total_clicks += 1.0
				true_items.append(target_item)
				rewards.append(reward)
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
		calculate_hit(sorted_list, topk, true_items, rewards, args.reward_click, total_reward, hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)
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

		hr_click, hr_purchase = round(hr_click, 4), round(hr_purchase, 4)
		ng_click, ng_purchase = round(ng_click, 4), round(ng_purchase, 4)

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