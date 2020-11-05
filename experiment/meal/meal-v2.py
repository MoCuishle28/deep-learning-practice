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

	# loading teacher

	teacher_models = args.teacher_models.split(',')
	GRU, CaserNet, NItNet, SASRec = None, None, None, None
	for model in teacher_models:
		if model == 'gru':
			gru_all_embeddings = initialize_embeddings(item_num, args.hidden_factor)
			GRU = GRUnetwork(args, hidden_size=args.hidden_factor, learning_rate=args.lr, dlr=0, item_num=item_num, state_size=state_size, embeddings=gru_all_embeddings)
		elif model == 'caser':
			caser_all_embeddings = initialize_embeddings(item_num, args.hidden_factor)
			CaserNet = Caser(args, hidden_size=args.hidden_factor, learning_rate=args.lr, dlr=0, item_num=item_num, state_size=state_size, embeddings=caser_all_embeddings)
		elif model == 'next':
			next_all_embeddings = initialize_embeddings(item_num, args.hidden_factor)
			NItNet = NextItNet(args, hidden_size=args.hidden_factor, learning_rate=args.lr, dlr=0, item_num=item_num, state_size=state_size, embeddings=next_all_embeddings)
		elif model == 'sas':
			sas_all_embeddings = initialize_embeddings(item_num, args.hidden_factor, state_size=state_size, pos=True)
			SASRec = SASRecnetwork(args, hidden_size=args.hidden_factor, learning_rate=args.lr, dlr=0, item_num=item_num, state_size=state_size, embeddings=sas_all_embeddings)
		else:
			info = f'wrong model:{model}...'
			print(info)
			logging.info(info)
		info = f'creating model:{model}'
		print(info)
		logging.info(info)
	ensemble = {'gru':GRU, 'caser':CaserNet, 'next':NItNet, 'sas':SASRec}
	saver = tf.train.Saver()


	student_model = None
	all_embeddings = initialize_embeddings(item_num, args.hidden_factor) if args.model != 'sas' else initialize_embeddings(item_num, args.hidden_factor, state_size=state_size, pos=True)
	if args.model == 'gru':		
		student_model = GRUnetwork(args, hidden_size=args.hidden_factor, learning_rate=args.lr, dlr=args.dlr, item_num=item_num, state_size=state_size, embeddings=all_embeddings, name='gru_student')
	elif args.model == 'caser':
		student_model = Caser(args, hidden_size=args.hidden_factor, learning_rate=args.lr, dlr=args.dlr, item_num=item_num, state_size=state_size, embeddings=all_embeddings, name='caser_student')
	elif args.model == 'next':
		student_model = NextItNet(args, hidden_size=args.hidden_factor, learning_rate=args.lr, dlr=args.dlr, item_num=item_num, state_size=state_size, embeddings=all_embeddings, name='next_student')
	elif args.model == 'sas':
		student_model = SASRecnetwork(args, hidden_size=args.hidden_factor, learning_rate=args.lr, dlr=args.dlr, item_num=item_num, state_size=state_size, embeddings=all_embeddings, name='sas_student')
	else:
		info = f'wrong model:{args.model}...'
		print(info)
		logging.info(info)
		assert 0>1
	info = f'creating student model:{args.model}'
	print(info)
	logging.info(info)

	total_step = 0
	max_ndcg_and_epoch = [[0, 0] for _ in args.topk.split(',')]	# (ng_inter, step)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.mem_ratio)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())

		# loading teacher_models
		ckpt = tf.train.get_checkpoint_state(f'multi/')
		saver.restore(sess, ckpt.model_checkpoint_path)
		info = f"loading Multi-Models."
		print(info)
		logging.info(info)
		# print('---------------------------------------')
		# print(sess.run(tf.get_default_graph().get_tensor_by_name('gru_teacher/fully_connected/weights:0')))	# debug

		num_rows = replay_buffer.shape[0]
		num_batches = int(num_rows/args.batch_size)
		for i in range(args.epoch):
			for j in range(num_batches):
				batch = replay_buffer.sample(n=args.batch_size).to_dict()
				state = list(batch['state'].values())
				len_state = list(batch['len_state'].values())
				# target = list(batch['action'].values())

				# aver soft label
				teacher_prob_sum = 0
				teacher_feature_sum = 0
				dis_input = []
				for k, model in ensemble.items():
					logits, prob = sess.run([model.output, model.predict_prob], feed_dict={model.inputs: state, 
						model.len_state: len_state, model.is_training: False})
					teacher_feature_sum += logits
					teacher_prob_sum += prob
					# dis_input.append(logits)	# every teacher's logits
				soft_label = teacher_prob_sum / len(ensemble.keys())	# aver probability
				# soft_label = teacher_feature_sum / len(ensemble.keys())	# aver logits

				dis_input.append(teacher_feature_sum / len(ensemble.keys())) # aver teacher's logits
				# DQN select teacher's soft label TODO
				# DDPG re-weight soft label 	  TODO

				# # train student
				# co-perform
				# prepare discriminator input
				dis_input = np.array(dis_input).reshape((-1, item_num))	# (batch, 26702) in RC15
				dis_label = [1 for _ in range(args.batch_size)]		# teacher label
				dis_label.extend([0 for _ in range(args.batch_size)])	# student label
				dis_label = np.array(dis_label).reshape(-1)			# (batch, 1)

				stu_loss, _, dis_loss, _ = sess.run([
					student_model.stu_loss, 
					student_model.stu_opt,
					student_model.dis_loss,
					student_model.dis_opt], 
					feed_dict={student_model.inputs: state, 
						student_model.len_state: len_state, 
						student_model.is_training: True,
						student_model.soft_label: soft_label,
						# discriminator
						student_model.teacher_logits: dis_input, 
						student_model.hard_label: dis_label})

				total_step += 1
				if (total_step % 200) == 0 or (total_step == 1):
					stu_loss, dis_loss = round(stu_loss.item(), 5), round(dis_loss.item(), 5)
					info = f'Epoch:[{i+1}/{args.epoch}] batch:[{total_step}] student loss: {stu_loss}, discriminator loss: {dis_loss}'
					print(info)
					logging.info(info)
				if (total_step >= args.start_eval) and (total_step % args.eval_interval == 0):
					t1 = time.time()
					if 'RC19' in args.base_data_dir:
						evaluate_RC19(args, student_model, sess, max_ndcg_and_epoch, total_step, logging)
					else:
						evaluate(args, student_model, sess, max_ndcg_and_epoch, total_step, logging)
					t2 = time.time()
					print(f'Time:{t2 - t1}')
					logging.info(f'Time:{t2 - t1}')


def parse_args():
	base_dir = '../data/'
	parser = argparse.ArgumentParser(description="Run Teacher Model.")
	parser.add_argument('--v', default="v")
	parser.add_argument('--teacher_models', default='gru,caser,next,sas')	# gru/caser/next/sas
	parser.add_argument('--model', default='gru')	# gru/caser/next/sas
	parser.add_argument('--mode', default='valid')
	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_data_dir', default=base_dir+'RC15')
	parser.add_argument('--topk', default='5,10,20')

	parser.add_argument('--epoch', type=int, default=1000)
	parser.add_argument('--eval_interval', type=int, default=2000)
	parser.add_argument('--start_eval', type=int, default=2000)
	parser.add_argument('--eval_batch', type=int, default=10)

	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--dlr', type=float, default=1e-4)
	parser.add_argument('--batch_size', type=int, default=256)

	# discriminator
	parser.add_argument('--dis_dropout_rate', type=float, default=0.5)
	parser.add_argument('--discriminator_layers', default='[1024,1024,1024]')
	parser.add_argument('--weight_decay', type=float, default=1e-5)

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
					filename = args.base_log_dir + f'meal-{args.model}-{args.v}-{str(time.time())}.log',
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