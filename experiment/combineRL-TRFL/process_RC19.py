import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def to_pickled_df(data_directory, **kwargs):
	for name, df in kwargs.items():
		df.to_pickle(os.path.join(data_directory, name + '.df'))

data_directory = '../data/RC19'

click_df = pd.read_csv(os.path.join(data_directory, 'train.csv'))
click_df = click_df[['session_id', 'timestamp', 'action_type','reference']]
click_df = click_df[(click_df['action_type'] == 'clickout item')].reset_index(drop=True)		# 1586585
# print(click_df)

click_df['valid_session'] = click_df.session_id.map(click_df.groupby('session_id')['reference'].size() > 2)
# click_df = click_df.dropna(axis=0, how='any')		# 存在 session_id 为空的
click_df = click_df.loc[click_df.valid_session].drop('valid_session', axis=1)
click_df = click_df.reset_index(drop=True)
# 752547 条数据, 158319 个 session ?
# print(click_df)
# print(click_df.groupby('session_id').size())

session_encoder = LabelEncoder()
click_df['user_session'] = session_encoder.fit_transform(click_df.session_id)
product_encoder = LabelEncoder()
click_df['product_id'] = product_encoder.fit_transform(click_df.reference)

click_df = click_df[['timestamp', 'product_id','user_session']]
click_df = click_df.sort_values(by=['user_session','timestamp']).reset_index(drop=True)
print(click_df)


# split data
total_ids = click_df.user_session.unique()
np.random.shuffle(total_ids)

fractions = np.array([0.8, 0.1, 0.1])
# split into 3 parts
train_ids, val_ids, test_ids = np.array_split(total_ids, (fractions[:-1].cumsum() * len(total_ids)).astype(int))

print('--train--')
print(train_ids)
print('--val--')
print(val_ids)
# print(test_ids)

train_sessions = click_df[click_df['user_session'].isin(train_ids)].reset_index(drop=True) 
val_sessions = click_df[click_df['user_session'].isin(val_ids)].reset_index(drop=True) 
test_sessions = click_df[click_df['user_session'].isin(test_ids)].reset_index(drop=True) 

# to_pickled_df(data_directory, click_df=click_df)
# to_pickled_df(data_directory, sampled_train=train_sessions)
# to_pickled_df(data_directory, sampled_val=val_sessions)
# to_pickled_df(data_directory,sampled_test=test_sessions)

print('---')
print(train_sessions)
print(val_sessions)
print(test_sessions)