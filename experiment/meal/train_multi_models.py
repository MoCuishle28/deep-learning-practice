import os
import logging
import argparse
import time
import datetime
import random

import tensorflow as tf
import trfl
import pandas as pd

from teachers import *
from utils import *


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
			GRU = GRUnetwork(args, hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num, state_size=state_size, embeddings=gru_all_embeddings)
		elif model == 'caser':
			caser_all_embeddings = initialize_embeddings(item_num, args.hidden_factor)
			CaserNet = Caser(args, hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num, state_size=state_size, embeddings=caser_all_embeddings)
		elif model == 'next':
			next_all_embeddings = initialize_embeddings(item_num, args.hidden_factor)
			NItNet = NextItNet(args, hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num, state_size=state_size, embeddings=next_all_embeddings)
		elif model == 'sas':
			sas_all_embeddings = initialize_embeddings(item_num, args.hidden_factor, state_size=state_size, pos=True)
			SASRec = SASRecnetwork(args, hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num, state_size=state_size, embeddings=sas_all_embeddings)
		else:
			info = f'wrong model:{model}...'
			print(info)
			logging.info(info)

		info = f'creating model:{model}'
		print(info)
		logging.info(info)
	ensemble = {'gru':GRU, 'caser':CaserNet, 'next':NItNet, 'sas':SASRec}

	saved_model = {k:False for k in teacher_models}
	loss_dict = {k:0 for k in teacher_models}
	total_step = 0
	max_ndcg_and_epoch_dict = {k:[[0, 0] for _ in args.topk.split(',')] for k in teacher_models}	# (ng_inter, step)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.mem_ratio)
	saver = tf.train.Saver()	# saver
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())
		num_rows = replay_buffer.shape[0]
		num_batches = int(num_rows/args.batch_size)
		for i in range(args.epoch):
			for j in range(num_batches):
				batch = replay_buffer.sample(n=args.batch_size).to_dict()
				state = list(batch['state'].values())
				len_state = list(batch['len_state'].values())
				target = list(batch['action'].values())

				for model in teacher_models:
					if saved_model[model]:
						print(f'model:{model} has saved...')
						continue
					ranking_model = ensemble.get(model)
					loss_dict[model], _ = sess.run([ranking_model.loss, ranking_model.opt],
									   feed_dict={ranking_model.inputs: state,
												  ranking_model.len_state: len_state,
												  ranking_model.target: target,
												  ranking_model.is_training:True})

				total_step += 1
				if (total_step % 200) == 0 or (total_step == 1):
					for k,v in loss_dict.items():
						loss_dict[k] = round(loss_dict[k].item(), 5)
					l1, l2, l3, l4 = loss_dict.get('gru'), loss_dict.get('caser'), loss_dict.get('next'), loss_dict.get('sas')
					info = f'batch:[{total_step}] loss: GRU->{l1}, Caser->{l2}, NItNet->{l3}, SASRec->{l4}'
					print(info)
					logging.info(info)

				if (total_step >= args.start_eval) and (total_step % args.eval_interval == 0):
					t1 = time.time()
					for model in teacher_models:
						if saved_model[model]:
							print(f'model:{model} has saved...')
							continue
						ranking_model = ensemble.get(model)
						info = f'\n\n===============================evaluating {model}==============================='
						print(info)
						logging.info(info)
						if 'RC19' in args.base_data_dir:
							evaluate_RC19(args, ranking_model, sess, max_ndcg_and_epoch_dict[model], total_step, logging)
						else:
							evaluate(args, ranking_model, sess, max_ndcg_and_epoch_dict[model], total_step, logging)
					t2 = time.time()
					print(f'Time:{t2 - t1}')
					logging.info(f'Time:{t2 - t1}')

				for model in teacher_models:
					if (total_step - max_ndcg_and_epoch_dict[model][0][1] >= 6000) and (total_step - max_ndcg_and_epoch_dict[model][1][1] >= 6000) and (total_step - max_ndcg_and_epoch[2][1] >= 6000):
						saved_model[model] = True
				if saved_model['gru'] and saved_model['caser'] and saved_model['next'] and saved_model['sas']:
					break

				# debug	
				# saver.save(sess, f"multi/multi-{args.v}", global_step=total_step) # save models
				# break
			if saved_model['gru'] and saved_model['caser'] and saved_model['next'] and saved_model['sas']:
				saver.save(sess, f"multi/multi-{args.v}", global_step=total_step) # save models
				print('save multi model')
				logging.info('save multi model')
				break


def parse_args():
	base_dir = '../data/'
	parser = argparse.ArgumentParser(description="Run Teacher Model.")
	parser.add_argument('--v', default="v")
	parser.add_argument('--models', default='gru,caser,next,sas')	# gru/caser/next/sas
	parser.add_argument('--mode', default='valid')
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_data_dir', default=base_dir+'RC15')
	parser.add_argument('--topk', default='5,10,20')

	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--eval_interval', type=int, default=2000)
	parser.add_argument('--start_eval', type=int, default=2000)
	parser.add_argument('--eval_batch', type=int, default=10)

	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--batch_size', type=int, default=256)

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

	parser.add_argument('--note', default='None...')
	parser.add_argument('--mem_ratio', type=float, default=0.2)
	parser.add_argument('--cuda', default='0')

	return parser.parse_args()

def init_log(args):
	if not os.path.exists(args.base_log_dir):
		os.makedirs(args.base_log_dir)
	start = datetime.datetime.now()
	logging.basicConfig(level = logging.INFO,
					filename = args.base_log_dir + 'multi-' + args.v + '-' + str(time.time()) + '.log',
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