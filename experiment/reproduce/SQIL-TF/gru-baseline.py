import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import trfl
import logging
import time
import datetime
import torch


def parse_args():
	parser = argparse.ArgumentParser(description="Run supervised GRU.")
	parser.add_argument('--v', default="v")
	parser.add_argument('--base_log_dir', default="baseline-log/")
	parser.add_argument('--data', default='../../data/kaggle-RL4REC')

	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--eval_interval', type=int, default=2000)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--hidden_factor', type=int, default=64)
	parser.add_argument('--r_click', type=float, default=0.2)
	parser.add_argument('--r_buy', type=float, default=1.0)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--seed', type=int, default=1)
	return parser.parse_args()


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

class GRUnetwork:
	def __init__(self, hidden_size, learning_rate, item_num, state_size):
		self.state_size = state_size
		self.learning_rate = learning_rate
		self.hidden_size = hidden_size
		self.item_num = int(item_num)

		all_embeddings = self.initialize_embeddings()

		self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
		self.len_state = tf.placeholder(tf.int32, [None],name='len_state')
		self.target = tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss

		self.input_emb = tf.nn.embedding_lookup(all_embeddings['state_embeddings'],self.inputs)

		gru_out, self.states_hidden= tf.nn.dynamic_rnn(
			tf.contrib.rnn.GRUCell(self.hidden_size),
			self.input_emb,
			dtype=tf.float32,
			sequence_length=self.len_state,
		)

		self.output = tf.contrib.layers.fully_connected(self.states_hidden,self.item_num,activation_fn=None,scope='fc')

		self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target,logits=self.output)
		self.loss = tf.reduce_mean(self.loss)
		self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


	def initialize_embeddings(self):
		all_embeddings = dict()
		state_embeddings= tf.Variable(tf.random_normal([self.item_num+1, self.hidden_size], 0.0, 0.01),
			name='state_embeddings')
		all_embeddings['state_embeddings']=state_embeddings
		return all_embeddings

def pad_history(itemlist, length, pad_item):
	if len(itemlist)>=length:
		return itemlist[-length:]
	if len(itemlist)<length:
		temp = [pad_item] * (length-len(itemlist))
		itemlist.extend(temp)
		return itemlist

def calculate_hit(sorted_list,topk,true_items,rewards,r_click,total_reward,hit_click,ndcg_click,hit_purchase,ndcg_purchase):
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

def evaluate(sess):
	eval_sessions=pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
	eval_ids = eval_sessions.session_id.unique()
	groups = eval_sessions.groupby('session_id')
	batch = 10
	evaluated=0
	total_clicks=0.0
	total_purchase = 0.0
	total_reward = [0, 0, 0, 0]
	hit_clicks=[0,0,0,0]
	ndcg_clicks=[0,0,0,0]
	hit_purchase=[0,0,0,0]
	ndcg_purchase=[0,0,0,0]
	while evaluated < len(eval_ids):
		states, len_states, actions, rewards = [], [], [], []
		for i in range(batch):
			if evaluated == len(eval_ids):
				break
			id = eval_ids[evaluated]
			group = groups.get_group(id)
			history = []
			for index, row in group.iterrows():
				state = list(history)
				len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
				state = pad_history(state,state_size,item_num)
				states.append(state)
				action = row['item_id']
				is_buy =row['is_buy']
				reward = reward_buy if is_buy == 1 else reward_click
				if is_buy==1:
					total_purchase+=1.0
				else:
					total_clicks+=1.0
				actions.append(action)
				rewards.append(reward)
				history.append(row['item_id'])
			evaluated+=1
		prediction=sess.run(GRUnet.output, feed_dict={GRUnet.inputs: states,GRUnet.len_state:len_states})
		# sorted_list=np.argsort(prediction)
		_, sorted_list = torch.topk(torch.tensor(prediction), max(topk), largest=False)
		sorted_list = sorted_list.numpy()
		calculate_hit(sorted_list,topk,actions,rewards,reward_click,total_reward,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase)
	print('#############################################################')
	print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
	for i in range(len(topk)):
		hr_click=hit_clicks[i]/total_clicks
		hr_purchase=hit_purchase[i]/total_purchase
		ng_click=ndcg_clicks[i]/total_clicks
		ng_purchase=ndcg_purchase[i]/total_purchase
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		print('cumulative reward @ %d: %f' % (topk[i],total_reward[i]))
		logging.info('cumulative reward @ %d: %f' % (topk[i],total_reward[i]))

		print('clicks hr ndcg @ %d : %f, %f' % (topk[i],hr_click,ng_click))
		logging.info('clicks hr ndcg @ %d : %f, %f' % (topk[i],hr_click,ng_click))

		print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
		logging.info('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
	print('#############################################################')
	logging.info('#############################################################')



if __name__ == '__main__':
	# Network parameters
	args = parse_args()
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)
	init_log(args)

	data_directory = args.data
	data_statis = pd.read_pickle(
		os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
	state_size = data_statis['state_size'][0]  # the length of history to define the state
	item_num = data_statis['item_num'][0]  # total number of items
	reward_click = args.r_click
	reward_buy = args.r_buy
	topk=[5,10,15,20]
	max_ndcg_and_epoch = [[0, 0, 0] for _ in args.topk.split(',')]	# (ng_click, ng_purchase, step)
	# save_file = 'pretrain-GRU/%d' % (hidden_size)

	tf.reset_default_graph()

	GRUnet = GRUnetwork(hidden_size=args.hidden_factor, learning_rate=args.lr,item_num=item_num,state_size=state_size)

	replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
	saver = tf.train.Saver()

	total_step=0
	with tf.Session() as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())
		# evaluate(sess)
		num_rows=replay_buffer.shape[0]
		num_batches=int(num_rows/args.batch_size)
		for i in range(args.epoch):
			for j in range(num_batches):
				batch = replay_buffer.sample(n=args.batch_size).to_dict()
				state = list(batch['state'].values())
				len_state = list(batch['len_state'].values())
				target=list(batch['action'].values())
				loss, _ = sess.run([GRUnet.loss, GRUnet.opt],
								   feed_dict={GRUnet.inputs: state,
											  GRUnet.len_state: len_state,
											  GRUnet.target: target})
				total_step+=1
				if (total_step == 1) or (total_step % 200 == 0):
					print("the loss in %dth batch is: %f" % (total_step, loss))
					logging.info("the loss in %dth batch is: %f" % (total_step, loss))
				if total_step % args.eval_interval == 0:
					t1 = time.time()
					evaluate(sess)
					t2 = time.time()
					print(f'Time:{t2 - t1}')
					logging.info(f'Time:{t2 - t1}')
		# saver.save(sess, save_file)