import os

import torch
import pandas as pd
import numpy as np


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


def evaluate(args, agent, sess, max_ndcg_and_epoch, total_step, logging):
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

		prediction = sess.run(agent.output, feed_dict={agent.inputs: states, agent.len_state: len_states})
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