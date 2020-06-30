import os
import pandas as pd
import tensorflow as tf


def pad_history(itemlist, length, pad_item):
	if len(itemlist) >= length:
		return itemlist[-length:]
	if len(itemlist) < length:
		temp = [pad_item] * (length - len(itemlist))
		itemlist.extend(temp)
		return itemlist

def to_pickled_df(data_directory, **kwargs):
	for name, df in kwargs.items():
		df.to_pickle(os.path.join(data_directory, name + '.df'))


if __name__ == '__main__':

	data_directory = '../data/Cosmetics-Shop'

	length = 10

	# reply_buffer = pd.DataFrame(columns=['state','action','reward','next_state','is_done'])
	click_df = pd.read_pickle(os.path.join(data_directory, 'click_df.df'))
	product_ids = click_df.product_id.unique()
	pad_item = len(product_ids)

	train_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_train.df'))
	groups = train_sessions.groupby('user_session')
	ids = train_sessions.user_session.unique()

	state, len_state, action, next_state, len_next_state, is_done = [], [], [], [], [], []

	for id in ids:
		group = groups.get_group(id)
		history = []
		for index, row in group.iterrows():
			s = list(history)
			len_state.append(length if len(s) >= length else 1 if len(s) == 0 else len(s))
			s = pad_history(s, length, pad_item)
			a = row['product_id']
			state.append(s)
			action.append(a)
			history.append(row['product_id'])
			next_s = list(history)
			len_next_state.append(length if len(next_s) >= length else 1 if len(next_s) == 0 else len(next_s))
			next_s = pad_history(next_s, length, pad_item)
			next_state.append(next_s)
			is_done.append(False)
		is_done[-1] = True

	dic = {'state':state, 'len_state':len_state, 'action':action, 'next_state':next_state, 'len_next_states':len_next_state, 'is_done':is_done}
	reply_buffer = pd.DataFrame(data=dic)
	to_pickled_df(data_directory, replay_buffer=reply_buffer)

	dic = {'state_size':[length],'item_num':[pad_item]}
	data_statis = pd.DataFrame(data=dic)
	to_pickled_df(data_directory, data_statis=data_statis)