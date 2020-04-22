import pandas as pd
import numpy as np

base_data_dir = '../../data/kaggle-RL4REC/'
# 相比源文件这个已经移除了交互次数小于2的 session
# events_pd = pd.read_csv(base_data_dir + 'sorted_events.csv')
# print(events_pd)
'''
		 timestamp  	session_id  item_id  is_buy
0        1442004589439           0    43511       0
1        1442004759591           0    54408       0
'''
# 0~70851	共 70852 个 item
# print(events_pd['item_id'].min(), events_pd['item_id'].max())
# s = set()
# for i in events_pd['item_id']:
# 	s.add(i)
# print(len(s))	

# val_sessions = pd.read_pickle(base_data_dir + 'sampled_val.df')
# print(val_sessions)
'''
 				timestamp  session_id  item_id  is_buy
1361687  1442004589439           0    43511       0
'''

# data_statis = pd.read_pickle(base_data_dir + 'data_statis.df')
# print(data_statis)		# state_size:10  item_num:70852


# replay_buffer = pd.read_pickle(base_data_dir + 'replay_buffer.df')
# print(replay_buffer)
'''
                            state  							len_state  ...  len_next_states  is_done
0       [70852, 70852, 70852, 70852, 70852, 70852, 708...          1  ...                1    False
1       [49432, 70852, 70852, 70852, 70852, 70852, 708...          1  ...                2    False
'''
# print(replay_buffer.shape)	# (988719, 7)

# batch = replay_buffer.sample(n=2, replace=False, random_state=1).to_dict()
# state = list(batch['state'].values())
# len_state = list(batch['len_state'].values())
# target=list(batch['action'].values())

# print('state:', state)
# print('len_state:', len_state)
# print('target:', target)

# print(type(replay_buffer))	# <class 'pandas.core.frame.DataFrame'>
# print(replay_buffer.shape)	# (988719, 7)

# i = 0
# state_list, next_state_list, action_list = [], [], []
# for idx, row in replay_buffer.iterrows():
# 	if row['len_state'] == 1:
# 		continue
# 	i += 1
# 	state_list.append(row['state'])
# 	next_state_list.append(row['next_state'])
# 	action_list.append(row['action'])

# state_list, next_state_list, action_list = np.array(state_list), np.array(next_state_list), np.array(action_list)
# print(state_list.shape)	# (680109, 10)
# print(i)	# 680109

# np.save(base_data_dir + 'state.npy', state_list)
# np.save(base_data_dir + 'next_state.npy', next_state_list)
# np.save(base_data_dir + 'action.npy', action_list)

# state = np.load(base_data_dir + 'state.npy')
# print(state.shape, state.dtype)
# print(state)

eval_sessions = pd.read_pickle(base_data_dir + 'sampled_val.df')
print(eval_sessions)
'''
 				timestamp  session_id  item_id  is_buy
1361687  	1442004589439           0    43511       0
'''

eval_ids = eval_sessions.session_id.unique()
print(eval_ids)		# [所有 sess id]
groups = eval_sessions.groupby('session_id')
for sid in eval_ids:
	group = groups.get_group(sid)
	history = []
	for index, row in group.iterrows():
		state = list(history)
		state = pad_history(state, 10, 70852)
