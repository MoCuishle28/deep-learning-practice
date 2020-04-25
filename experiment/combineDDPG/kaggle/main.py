import os
import pickle
import argparse
import random
import logging
import datetime
import time

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np

from seqddpg import DDPG
from seqddpg import Transition
from seqddpg import ReplayMemory
from seqddpg import OUNoise
from seqsac import SAC
from models import MLP, Predictor
from evaluate import Evaluation


class Run(object):
	def __init__(self, args, agent, predictor, device):
		self.device = device
		self.args = args
		self.agent = agent
		layers = [int(x) for x in args.a_layers.split(',')]
		self.ounoise = OUNoise(layers[-1]) if args.agent == 'ddpg' else None
		self.predictor = predictor
		self.memory = ReplayMemory(args.memory_size)

		self.evaluate = Evaluation(args, device, agent, predictor)
		self.build_data_loder()


	def build_data_loder(self):
		state_list = torch.tensor(np.load(self.args.base_data_dir + 'state.npy'), dtype=torch.float32, device=self.device)
		next_state_list = torch.tensor(np.load(self.args.base_data_dir + 'next_state.npy'), dtype=torch.float32, device=self.device)
		action_list = torch.tensor(np.load(self.args.base_data_dir + 'action.npy'), dtype=torch.long, device=self.device)
		dataset = Data.TensorDataset(state_list, next_state_list, action_list)
		self.data_loader = Data.DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=True if self.args.shuffle == 'y' else False)


	def fill_replay(self, data):
		self.agent.on_eval()	# 切换为评估模式
		self.predictor.on_train()
		# [512, 10] [512, 10] [512]
		batch_state, batch_next_state, batch_target = data
		with torch.no_grad():
			batch_action = self.agent.select_action(batch_state, action_noise=self.ounoise)
		model_loss, reward_list = self.predictor.train(batch_action.detach(), batch_target)

		for state, action, next_state, r in zip(batch_state, batch_action, batch_next_state, reward_list):
			next_state = next_state if r != 0 else state
			reward = torch.tensor([r * self.args.alpha], dtype=torch.float32, device=self.device)
			state, next_state, action = state.view(1, -1, 1), next_state.view(1, -1, 1), action.view(1, -1)
			self.memory.push(state, action, next_state, reward)

		if self.args.reset == 'y':
			self.ounoise.reset()	# reset noise vector
		aver_reward = torch.tensor(reward_list, dtype=torch.float32, device=self.device).mean() * self.args.alpha
		return model_loss, aver_reward.item()


	def train(self):
		model_loss_list, average_reward_list = [], []
		hr_list, ndcg_list, precision_list = [], [], []
		max_ndcg, max_ndcg_epoch = 0, 0

		for epoch in range(self.args.epoch):
			for i_batch, data in enumerate(self.data_loader):
				model_loss, aver_reward = self.fill_replay(data)

				transitions = self.memory.sample(self.args.batch_size)
				batch = Transition(*zip(*transitions))
				self.agent.on_train()	# 切换为训练模式 (因为包含 BN\LN)
				policy_loss, critic_loss, value_loss = self.agent.update_parameters(batch)
				policy_loss, model_loss, aver_reward = round(policy_loss, 3), round(model_loss, 3), round(aver_reward, 3)
				critic_loss = round(critic_loss, 3) if critic_loss != None else None
				value_loss = round(value_loss, 3) if value_loss != None else None

				if i_batch % 200 == 0:
					info = f'epoch:{epoch+1}/{self.args.epoch} @{self.args.topk} i_batch:{i_batch}, Average Reward:{aver_reward}, Model LOSS:{model_loss}'
					print(info, end = ', ')
					print(f'critic loss:{critic_loss}, policy loss:{policy_loss}, value loss:{value_loss}')
					logging.info(info)
					average_reward_list.append(aver_reward), model_loss_list.append(model_loss)

			if (epoch + 1) >= self.args.start_save and (epoch + 1) % self.args.save_interval == 0:
				self.agent.save_model(version=self.args.v, epoch=epoch)
				self.predictor.save(self.args.v, epoch=epoch)
				info = f'Saving version:{args.v}_{epoch} models'
				print(info)
				logging.info(info)

			if ((epoch + 1) >= self.args.start_eval) and ((epoch + 1) % self.args.eval_interval == 0):
				self.agent.on_eval()
				self.predictor.on_eval()
				t1 = time.time()
				with torch.no_grad():
					hr, ndcg, precs = self.evaluate.eval()
				hr, ndcg, precs = round(hr, 5), round(ndcg, 5), round(precs, 5)
				t2 = time.time()
				if ndcg > max_ndcg:
					max_ndcg = ndcg
					max_ndcg_epoch = epoch
				else:
					# TODO lr decay
					pass
				info = f'[{self.args.mode}]@{self.args.topk} HR:{hr}, NDCG:{ndcg}, Precision:{precs}, Time:{t2 - t1}, Current Max NDCG:{max_ndcg} (epoch:{max_ndcg_epoch})'
				print(info)
				logging.info(info)
				hr_list.append(hr), ndcg_list.append(ndcg), precision_list.append(precs)

		self.evaluate.plot_result(precision_list, hr_list, ndcg_list, model_loss_list, average_reward_list)


def init_log(args):
	if not os.path.exists(args.base_log_dir):
		os.makedirs(args.base_log_dir)
	if not os.path.exists(args.base_pic_dir):
		os.makedirs(args.base_pic_dir)
	start = datetime.datetime.now()
	logging.basicConfig(level = logging.INFO,
					filename = args.base_log_dir + args.v + '-' + str(time.time()) + '.log',
					filemode = 'a',		# 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
					# a是追加模式，默认如果不写的话，就是追加模式
					)
	logging.info('start! '+str(start))
	logging.info('Parameter:')
	logging.info(str(args))
	logging.info('\n-------------------------------------------------------------\n')


def main(args, device):
	agent = None
	info = f'RL Agent is {args.agent}'
	print(info)
	logging.info(info)
	if args.agent == 'ddpg':
		agent = DDPG(args, device)
	elif args.agent == 'sac':
		agent = SAC(args, device)

	predictor_model = None
	layers = [int(x) for x in args.a_layers.split(',')]
	actor_output = layers[0]
	if args.predictor == 'mlp':
		predictor_model = MLP(args, device)
		print('predictor_model is MLP.')
		logging.info('predictor_model is MLP.')
	elif args.predictor == '':
		# TODO
		pass
		
	predictor = Predictor(args, predictor_model, device)

	# 加载模型
	if args.load == 'y':
		agent.load_model(version=args.load_version, epoch=args.load_epoch)
		predictor.load(args.load_version, epoch=args.load_epoch)
		info = f'Loading version:{args.load_version}_{args.load_epoch} models'
		print(info)
		logging.info(info)

	run = Run(args, agent, predictor, device)
	run.train()

	# 保存模型
	if args.save == 'y':
		run.agent.save_model(version=args.v, epoch='final')
		predictor.save(args.v, epoch='final')
		info = f'Saving version:{args.v}_final models'
		print(info)
		logging.info(info)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--v', default="v")
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--mode', default='valid')	# valid/test

	parser.add_argument('--base_log_dir', default="log/")
	parser.add_argument('--base_pic_dir', default="pic/")
	parser.add_argument('--base_data_dir', default='../../data/kaggle-RL4REC/')

	parser.add_argument('--epoch', type=int, default=2000)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--start_save', type=int, default=0)
	parser.add_argument('--save_interval', type=int, default=100)			# 多少个 epoch 保存一次模型
	parser.add_argument('--start_eval', type=int, default=0)
	parser.add_argument('--eval_interval', type=int, default=100)		# 多少个 epoch 评估一次
	parser.add_argument('--shuffle', default='y')
	# RL setting
	parser.add_argument('--reset', default='n')
	parser.add_argument('--memory_size', type=int, default=10000)
	parser.add_argument('--predictor', default='mlp')
	parser.add_argument('--reward', default='ndcg')
	parser.add_argument('--alpha', type=float, default=1)	# raw reward 乘以的倍数(试图放大 reward 加大训练幅度)

	parser.add_argument('--predictor_optim', default='adam')
	parser.add_argument('--actor_optim', default='adam')
	parser.add_argument('--critic_optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)	# sgd 时
	parser.add_argument('--weight_decay', type=float, default=1e-4)		# regularization
	parser.add_argument('--layer_trick', default='none')			# bn/ln/none
	parser.add_argument('--dropout', type=float, default=0.0)	# dropout (BN 可以不需要)
	# save model 的名字为 --v; load model 名字为 --load_version + _ + --load_epoch
	parser.add_argument('--save', default='y')
	parser.add_argument('--load', default='n')
	parser.add_argument('--load_version', default='v')
	parser.add_argument('--load_epoch', default='final')
	parser.add_argument('--show', default='n')
	# init weight
	parser.add_argument('--init_std', type=float, default=0.1)
	# seq model
	parser.add_argument('--seq_hidden_size', type=int, default=128)
	parser.add_argument('--seq_layer_num', type=int, default=1)
	parser.add_argument('--seq_output_size', type=int, default=128)
	# agent
	parser.add_argument('--agent', default='sac')
	parser.add_argument('--gamma', type=float, default=0.99)

	parser.add_argument("--actor_lr", type=float, default=1e-4)
	parser.add_argument("--critic_lr", type=float, default=1e-3)
	parser.add_argument('--value_lr', type=float, default=1e-3)
	parser.add_argument('--actor_tau', type=float, default=0.1)
	parser.add_argument('--critic_tau', type=float, default=0.1)
	parser.add_argument('--a_act', default='relu')
	parser.add_argument('--c_act', default='relu')
	parser.add_argument('--v_act', default='relu')
	parser.add_argument('--a_layers', default='128,64')	# seq_output_size, ...
	parser.add_argument('--c_layers', default='192,1')	# seq_output_size + actor_output, ...
	parser.add_argument('--v_layers', default='128,1')
	# sac
	parser.add_argument('--mean_lambda', type=float, default=1e-3)
	parser.add_argument('--std_lambda', type=float, default=1e-3)
	parser.add_argument('--z_lambda', type=float, default=0.0)
	parser.add_argument('--log_std_min', type=float, default=-20)
	parser.add_argument('--log_std_max', type=float, default=2)
	# predictor
	parser.add_argument("--predictor_lr", type=float, default=1e-3)
	# embedding
	parser.add_argument('--max_iid', type=int, default=70851)	# 0~70851
	parser.add_argument('--i_emb_dim', type=int, default=128)
	# MLP
	parser.add_argument('--mlp_act', default='relu')
	parser.add_argument('--mlp_layers', default='64,70851')

	args = parser.parse_args()
	init_log(args)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tmp = 'cuda' if torch.cuda.is_available() else 'cpu'

	# 保持可复现
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)

	print('device:{}'.format(device))
	main(args, device)