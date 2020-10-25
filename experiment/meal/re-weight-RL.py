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


def sample_data(args, replay_buffer):
	batch = replay_buffer.sample(n=args.batch_size).to_dict()
	state = list(batch['state'].values())
	len_state = list(batch['len_state'].values())
	next_state = list(batch['next_state'].values())
	len_next_states = list(batch['len_next_states'].values())
	target_items = list(batch['action'].values())
	is_done = list(batch['is_done'].values())
	return state, len_state, next_state, len_next_states, target_items, is_done


def trans_state(rewards, state, next_state, len_state, len_next_states):
	true_next_state, true_next_state_len = [], []
	for r, s, s_, sl, sl_ in zip(rewards, state, next_state, len_state, len_next_states):
		if r == 0:
			true_next_state.append(s)
			true_next_state_len.append(sl)
		else:
			true_next_state.append(s_)
			true_next_state_len.append(sl_)
	return true_next_state, true_next_state_len


def get_NDCG(args, logits, target_items):
	logits = torch.tensor(logits)
	_, rankings = logits.topk(args.reward_top)
	rankings = rankings.tolist()	# (batch, topk)
	rewards = []
	for target_iid, rec_list in zip(target_items, rankings):
		ndcg = 0.0
		for i, iid in enumerate(rec_list):
			if iid == target_iid:
				ndcg = 1.0 / np.log2(i + 2.0).item()
				break
		rewards.append(ndcg)
	return rewards


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

	# create RL agent
	main_agent, target_agent = None, None
	state_size = sum([model.state_size for model in ensemble.values()])
	if args.rl == 'ddpg':
		main_agent = DDPG(args, state_size=state_size, teacher_num=len(ensemble.keys()), max_action=args.max_action, name='main_agent')
		target_agent = DDPG(args, state_size=state_size, teacher_num=len(ensemble.keys()), max_action=args.max_action, name='target_agent')
		target_network_update_ops = trfl.update_target_variables(target_agent.get_qnetwork_variables(), 
			main_agent.get_qnetwork_variables(), tau=args.tau)
		copy_weight = trfl.update_target_variables(target_agent.get_qnetwork_variables(), 
			main_agent.get_qnetwork_variables(), tau=1.0)
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

	saved_model = {k:False for k in teacher_models}
	saved_model['rl'] = False
	loss_dict = {k:0 for k in teacher_models}
	total_step = 0
	max_ndcg_and_epoch_dict = {k:[[0, 0] for _ in args.topk.split(',')] for k in teacher_models}	# (ng_inter, step)
	max_ndcg_and_epoch_dict['rl'] = [[0, 0] for _ in args.topk.split(',')]
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.mem_ratio)
	saver = tf.train.Saver()	# saver
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(tf.global_variables_initializer())		# Initialize variables
		sess.run(copy_weight)		# copy weights

		num_rows = replay_buffer.shape[0]
		num_batches = int(num_rows/args.batch_size)
		discount = [args.gamma] * args.batch_size
		for i in range(args.epoch):
			for j in range(num_batches):
				state, len_state, next_state, len_next_states, target_items, is_done = sample_data(args, replay_buffer)

				rl_input_state, rl_input_next_state, logits_list = [], [], []
				for model in teacher_models:
					ranking_model = ensemble.get(model)
					if saved_model[model]:
						logits, s = sess.run([ranking_model.output, ranking_model.state_hidden],
						   feed_dict={ranking_model.inputs: state,
						   			ranking_model.len_state: len_state,
									ranking_model.is_training:False})
					else:
						logits, s, loss_dict[model], _ = sess.run([ranking_model.output,
							ranking_model.state_hidden,
							ranking_model.loss, 
							ranking_model.opt],
						   feed_dict={ranking_model.inputs: state, 
						   			ranking_model.len_state: len_state,
									ranking_model.target: target_items,
									ranking_model.is_training:True})
					# s_ = sess.run(ranking_model.state_hidden,
					# 	   feed_dict={ranking_model.inputs: next_state,
					# 	   			ranking_model.len_state: len_next_states,
					# 				ranking_model.is_training:False})
					rl_input_state.append(s)
					logits_list.append(logits)
					# rl_input_next_state.append(s_)

				rl_input_state = np.concatenate(rl_input_state, axis=-1).reshape((-1, state_size)) # (batch, state_size)
				if args.rl == 'ddpg':
					actions = sess.run(main_agent.actor_out_, feed_dict={
						main_agent.inputs: rl_input_state, main_agent.is_training: False})
					noise = np.random.normal(0, args.noise_var, size=main_agent.teacher_num).clip(-args.noise_clip, args.noise_clip)
					actions = (actions + noise).clip(-args.max_action, args.max_action)
					# (batch, 112(teacher_num*hidden_size))
					actor_weights = torch.nn.functional.softmax(torch.tensor(actions), dim=-1).numpy()

					# re-weighting
					final_logits = 0
					for idx, logits in enumerate(logits_list):
						weight = actor_weights[:, idx].reshape((-1, 1))
						final_logits += (weight * logits)

					if args.reward == 'ndcg':
						rewards = get_NDCG(args, final_logits, target_items)
					else:
						pass

					# trans state
					rl_next_state, rl_next_state_len = trans_state(rewards, state, next_state, len_state, len_next_states)
					for model in teacher_models:
						ranking_model = ensemble.get(model)
						s_ = sess.run(ranking_model.state_hidden,
						   feed_dict={ranking_model.inputs: rl_next_state,
						   			ranking_model.len_state: rl_next_state_len,
									ranking_model.is_training:False})
						rl_input_next_state.append(s_)
					rl_input_next_state = np.concatenate(rl_input_next_state, axis=-1).reshape((-1, state_size))

					target_v = sess.run(target_agent.critic_output, feed_dict={
						target_agent.inputs: rl_input_next_state,
						target_agent.is_training: False})
					target_v = target_v.squeeze()
					for index in range(args.batch_size):
						if is_done[index]:
							target_v[index] = 0.0

					critic_loss, _ = sess.run([main_agent.critic_loss, 
						main_agent.critic_optim], 
						feed_dict={main_agent.inputs:rl_input_state,
						main_agent.actor_out_: actions, 
						main_agent.reward: rewards,
						main_agent.discount: discount,
						main_agent.target: target_v,
						main_agent.is_training: True})
					actor_loss, _ = sess.run([main_agent.actor_loss, main_agent.actor_optim],
						feed_dict={main_agent.inputs: rl_input_state,
						main_agent.is_training: True})
					sess.run(target_network_update_ops)		# update target net
				elif args.rl == 'dqn':
					# TODO
					pass
				else:
					assert main_agent != None

				total_step += 1
				if (total_step % 200) == 0 or (total_step == 1):
					for k,v in loss_dict.items():
						if not saved_model[k]:
							loss_dict[k] = round(loss_dict[k].item(), 3)
					l1, l2, l3, l4 = loss_dict.get('gru'), loss_dict.get('caser'), loss_dict.get('next'), loss_dict.get('sas')
					aver_reward = round(np.array(rewards).mean().item(), 3)
					actor_loss, critic_loss = round(actor_loss.item(), 4), round(critic_loss.item(), 4)
					info = f'Epoch:[{i+1}/{args.epoch}] batch:[{total_step}] loss: GRU->{l1}, Caser->{l2}, NItNet->{l3}, SASRec->{l4}, Aver Reward:{aver_reward}, ALoss:{actor_loss}, CLoss:{critic_loss}'
					print(info)
					logging.info(info)

				if (total_step >= args.start_eval) and (total_step % args.eval_interval == 0):
					t1 = time.time()
					for model in teacher_models:
						if saved_model[model]:
							print(f'\nmodel:{model} has saved...')
							logging.info(f'\nmodel:{model} has saved...')
							continue
						ranking_model = ensemble.get(model)
						info = f'\n\n===============================evaluating {model}==============================='
						print(info)
						logging.info(info)
						if 'RC19' in args.base_data_dir:
							evaluate_RC19(args, ranking_model, sess, max_ndcg_and_epoch_dict[model], total_step, logging)
						else:
							evaluate(args, ranking_model, sess, max_ndcg_and_epoch_dict[model], total_step, logging)
					info = f'\n\n===============================evaluating withRL==============================='
					print(info)
					logging.info(info)
					if args.rl == 'ddpg':
						evaluate_reWeight(args, ensemble, main_agent, sess, max_ndcg_and_epoch_dict['rl'], total_step, logging, RC19=(True if 'RC19' in args.base_data_dir else False))
					else:
						# TODO
						pass
					t2 = time.time()
					print(f'Time:{t2 - t1}')
					logging.info(f'Time:{t2 - t1}')

				for model in teacher_models:
					if (total_step >= args.start_eval) and (total_step - max_ndcg_and_epoch_dict[model][0][1] >= 6000) and (total_step - max_ndcg_and_epoch_dict[model][1][1] >= 6000) and (total_step - max_ndcg_and_epoch_dict[model][2][1] >= 6000):
						saved_model[model] = True
				if (total_step >= args.start_eval) and (total_step - max_ndcg_and_epoch_dict['rl'][0][1] >= 6000) and (total_step - max_ndcg_and_epoch_dict['rl'][1][1] >= 6000) and (total_step - max_ndcg_and_epoch_dict['rl'][2][1] >= 6000):
					saved_model['rl'] = True
					break

				# debug	
				# saver.save(sess, f"rl/rl-{args.v}", global_step=total_step) # save models
				# break
			if saved_model['rl']:
				saver.save(sess, f"rl/rl-{args.v}", global_step=total_step) # save models
				print('save rl-multi model')
				logging.info('save rl-multi model')
				break


def parse_args():
	base_dir = '../data/'
	parser = argparse.ArgumentParser(description="Run Teacher Model.")
	parser.add_argument('--v', default="v")
	parser.add_argument('--models', default='gru,caser,next,sas')	# gru/caser/next/sas
	parser.add_argument('--rl', default='ddpg')	# ddpg/dqn
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

	# RL
	parser.add_argument('--tau', type=float, default=0.001)
	parser.add_argument('--gamma', type=float, default=0.5)
	parser.add_argument('--weight_decay', default=1e-5, type=float)
	parser.add_argument('--reward_top', type=int, default=20)
	parser.add_argument('--reward', default='ndcg')
	# DDPG
	parser.add_argument('--noise_var', type=float, default=0.1)
	parser.add_argument('--noise_clip', type=float, default=0.5)
	parser.add_argument('--rl_dropout_rate', type=float, default=0.5)
	parser.add_argument('--actor_layers', default="[]")
	parser.add_argument('--critic_layers', default="[]")
	parser.add_argument('--max_action', type=float, default=1.0)
	parser.add_argument('--alr', type=float, default=3e-4)
	parser.add_argument('--clr', type=float, default=3e-4)
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
					filename = args.base_log_dir + 'RL-multi-' + args.v + '-' + str(time.time()) + '.log',
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