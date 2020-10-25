import os
import logging
import argparse
import time
import datetime
import random

import tensorflow as tf
import trfl
import pandas as pd
import torch

from teachers import *
from utils import *
from rlagent import *


def main(args):
	data_directory = args.base_data_dir
	data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
	state_size = data_statis['state_size'][0]  # the length of history to define the state
	item_num = data_statis['item_num'][0]  # total number of items

	replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))

	tf.reset_default_graph()

	teacher_models = args.models.split(',')
	GRU, CaserNet, NItNet, SASRec = None, None, None, None
	for model in teacher_models:
		if model == 'gru':
			gru_all_embeddings = initialize_embeddings(item_num, args.hidden_factor)
			GRU = GRUnetwork(args, hidden_size=args.hidden_factor, learning_rate=0, dlr=0, item_num=item_num, state_size=state_size, embeddings=gru_all_embeddings)
		elif model == 'caser':
			caser_all_embeddings = initialize_embeddings(item_num, args.hidden_factor)
			CaserNet = Caser(args, hidden_size=args.hidden_factor, learning_rate=0, dlr=0, item_num=item_num, state_size=state_size, embeddings=caser_all_embeddings)
		elif model == 'next':
			next_all_embeddings = initialize_embeddings(item_num, args.hidden_factor)
			NItNet = NextItNet(args, hidden_size=args.hidden_factor, learning_rate=0, dlr=0, item_num=item_num, state_size=state_size, embeddings=next_all_embeddings)
		elif model == 'sas':
			sas_all_embeddings = initialize_embeddings(item_num, args.hidden_factor, state_size=state_size, pos=True)
			SASRec = SASRecnetwork(args, hidden_size=args.hidden_factor, learning_rate=0, dlr=0, item_num=item_num, state_size=state_size, embeddings=sas_all_embeddings)
		else:
			info = f'wrong model:{model}...'
			print(info)
			logging.info(info)

		info = f'creating model:{model}'
		print(info)
		logging.info(info)
	ensemble = {'gru':GRU, 'caser':CaserNet, 'next':NItNet, 'sas':SASRec}

	# create RL agent
	main_agent, target_agent = None, None
	state_size = sum([model.state_size for model in ensemble.values()])
	if args.rl == 'ddpg':
		main_agent = DDPG(args, state_size=state_size, teacher_num=len(ensemble.keys()), max_action=args.max_action, name='main_agent')
		target_agent = DDPG(args, state_size=state_size, teacher_num=len(ensemble.keys()), max_action=args.max_action, name='target_agent')
	elif args.rl == 'dqn':
		pass
	else:
		info = f'wrong RL agent:{args.rl}...'
		print(info)
		logging.info(info)
		assert main_agent != None
	info = f'creating RL agent:{args.rl}'
	print(info)
	logging.info(info)

	total_step = 0
	max_ndcg_and_epoch_dict = {k:[[0, 0] for _ in args.topk.split(',')] for k in teacher_models}	# (ng_inter, step)
	max_ndcg_and_epoch_dict['rl'] = [[0, 0] for _ in args.topk.split(',')]
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.mem_ratio)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		saver = tf.train.Saver()

		ckpt = tf.train.get_checkpoint_state(f'rl/')
		saver.restore(sess, ckpt.model_checkpoint_path)
		info = f"loading with RL-Models."
		print(info)
		logging.info(info)

		info = f'\n\n===============================evaluating withRL==============================='
		print(info)
		logging.info(info)
		if args.rl == 'ddpg':
			evaluate_reWeight(args, ensemble, main_agent, sess, max_ndcg_and_epoch_dict['rl'], 
				total_step, logging, RC19=(True if 'RC19' in args.base_data_dir else False))
		else:
			# TODO
			pass


def parse_args():
	base_dir = '../data/'
	parser = argparse.ArgumentParser(description="Run Teacher Model.")
	parser.add_argument('--v', default="v")
	parser.add_argument('--models', default='gru,caser,next,sas')	# gru/caser/next/sas
	parser.add_argument('--rl', default='ddpg')	# ddpg/dqn
	parser.add_argument('--mode', default='test')
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_data_dir', default=base_dir+'RC15')
	parser.add_argument('--topk', default='5,10,20')

	parser.add_argument('--eval_batch', type=int, default=10)

	# GRU4Rec/Caser/NItNet/SASRec
	parser.add_argument('--hidden_factor', type=int, default=16,
						help='Number of hidden factors, i.e., embedding size.')
	parser.add_argument('--dropout_rate', default=0.1, type=float)
	# parser.add_argument('--weight_decay', default=1e-4, type=float)

	# GRU4Rec
	parser.add_argument('--layer_trick', default='ln')
	# Caser
	parser.add_argument('--num_filters', type=int, default=16,
						help='Number of filters per filter size (default: 128)')
	parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
						help='Specify the filter_size')
	# SASRec
	parser.add_argument('--num_heads', default=1, type=int)
	parser.add_argument('--num_blocks', default=1, type=int)

	# RL
	parser.add_argument('--weight_decay', default=1e-5, type=float)
	# DDPG
	parser.add_argument('--noise_var', type=float, default=0.1)
	parser.add_argument('--noise_clip', type=float, default=0.5)
	parser.add_argument('--rl_dropout_rate', type=float, default=0.5)
	parser.add_argument('--actor_layers', default="[]")
	parser.add_argument('--critic_layers', default="[]")
	parser.add_argument('--alr', default=3e-4)
	parser.add_argument('--clr', default=3e-4)
	parser.add_argument('--max_action', type=float, default=1.0)
	# DQN
	# TODO
	
	parser.add_argument('--note', default='None...')
	parser.add_argument('--mem_ratio', type=float, default=0.2)
	parser.add_argument('--cuda', default='0')

	return parser.parse_args()

def init_log(args):
	if not os.path.exists(args.base_log_dir):
		os.makedirs(args.base_log_dir)
	start = datetime.datetime.now()
	logging.basicConfig(level = logging.INFO,
					filename = args.base_log_dir + 'RL-test-' + args.v + '-' + str(time.time()) + '.log',
					filemode = 'a',
					)
	print('start! '+str(start))
	logging.info('start! '+str(start))
	logging.info('Parameter:')
	logging.info(str(args))
	logging.info('\n-------------------------------------------------------------\n')

if __name__ == '__main__':
	args = parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	init_log(args)
	main(args)