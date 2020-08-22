import os
import pandas as pd

def count_len(sampled_data, sid_to_len):
	for _, row in sampled_data.iterrows():
		if row['session_id'] in sid_to_len:
			sid_to_len[row['session_id']] += 1
		else:
			sid_to_len[row['session_id']] = 1

def statistic(sid_to_len):
	cnt = [0 for _ in range(7)]
	for sid, size in sid_to_len.items():
		if size < 10:
			cnt[0] += 1
		elif size >= 10 and size <20:
			cnt[1] += 1
		elif size >= 20 and size < 40:
			cnt[2] += 1
		elif size >= 40 and size < 60:
			cnt[3] += 1
		elif size >= 60 and size < 80:
			cnt[4] += 1
		elif size >= 80 and size < 100:
			cnt[5] += 1
		elif size >= 100:
			cnt[6] += 1
	return cnt

def aver_len(sid_to_len):
	sum_len = 0
	for k, v in sid_to_len.items():
		sum_len += v
	print(sum_len, sum_len / len(sid_to_len.keys()))


# data_directory = '../../../data/RC15'
# # replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
# sampled_train = pd.read_pickle(os.path.join(data_directory, 'sampled_train.df'))
# # print(sampled_train)

# sid_to_len = {}
# count_len(sampled_train, sid_to_len)
# print(min(sid_to_len.values()), max(sid_to_len.values()), len(sid_to_len.keys()))	# 3 198

# sampled_val = pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
# count_len(sampled_val, sid_to_len)
# print(min(sid_to_len.values()), max(sid_to_len.values()), len(sid_to_len.keys()))	# 

# sampled_test = pd.read_pickle(os.path.join(data_directory, 'sampled_test.df'))
# count_len(sampled_test, sid_to_len)
# print(min(sid_to_len.values()), max(sid_to_len.values()), len(sid_to_len.keys()))	# 

# aver_len(sid_to_len)	# 1154911 5.774555

# cnt = statistic(sid_to_len)
# # [195698, 3727, 418, 109, 32, 16]
# # [88.57%, 9.27%, 1.86%, 0.2%, 0.05%, 0.016%, 0.008%]
# # [177147, 18551, 3727, 418, 109, 32, 16]
# print(cnt)

# assert 0>1

# data_directory = '../../../data/kaggle-RL4REC'
# # replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
# sampled_train = pd.read_pickle(os.path.join(data_directory, 'sampled_train.df'))
# # print(sampled_train)

# sid_to_len = {}
# count_len(sampled_train, sid_to_len)
# print(min(sid_to_len.values()), max(sid_to_len.values()), len(sid_to_len.keys()))	# 1 7018

# sampled_val = pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
# count_len(sampled_val, sid_to_len)
# print(min(sid_to_len.values()), max(sid_to_len.values()), len(sid_to_len.keys()))	# 

# sampled_test = pd.read_pickle(os.path.join(data_directory, 'sampled_test.df'))
# count_len(sampled_test, sid_to_len)
# print(min(sid_to_len.values()), max(sid_to_len.values()), len(sid_to_len.keys()))	# 

# aver_len(sid_to_len)	# 1233949 6.310984840735665

# cnt = statistic(sid_to_len)
# # [89.16%, 7.78%, 2.16%, 0.44%, 0.16%, 0.07%, 0.2%]
# print(cnt)	# [174329, 15205, 4288, 877, 312, 131, 382]	195,533

# assert 0>1

data_directory = '../../../data/RC19'
# click_df = pd.read_pickle(os.path.join(data_directory, 'click_df.df'))
# print(click_df)

# sid_to_len = {}
# for _, row in click_df.iterrows():
# 	if row['user_session'] in sid_to_len:
# 		sid_to_len[row['user_session']] += 1
# 	else:
# 		sid_to_len[row['user_session']] = 1

# cnt = statistic(sid_to_len)
# # [94.22%, 5.1%, 0.63%, 0.03%, 0.006%, 0.002%, 0%]
# print(cnt)	# [149170, 8079, 997, 59, 10, 4, 0]

# click_df = pd.read_pickle(os.path.join(data_directory, 'click_df.df'))

# sid_to_len = {}
# for _, row in click_df.iterrows():
# 	if row['user_session'] in sid_to_len:
# 		sid_to_len[row['user_session']] += 1
# 	else:
# 		sid_to_len[row['user_session']] = 1

# sum_len = 0
# for k, v in sid_to_len.items():
# 	sum_len += v

# print(sum_len)
# print(sum_len / len(sid_to_len.keys()))	# 752548 4.753365041466911