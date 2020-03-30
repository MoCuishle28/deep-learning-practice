import sys
sys.path.append("..")
import os
import argparse
import logging
import multiprocessing
import time
import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from myMF import Predictor, MF
from evaluate import Evaluate
from utils import Utils


class Run(object):
	def __init__(self, args, device, predictor):
		super(Run, self).__init__()
		self.args = args
		self.device = device
		self.predictor = predictor

		self.valid_data = np.load(args.base_data_dir + 'valid_data.npy').astype(np.float32)
		self.test_data = np.load(args.base_data_dir + 'test_data.npy').astype(np.float32)
		self.evaluate = Evaluate(args, device, agent, predictor, None, self.valid_data, self.test_data, env)

	def eval(self):
		pass


def main(args, device):
	model = None
	predictor = None

	info = f'Predictor is {args.predictor}.'
	print(info)
	logging.info(info)
	if args.predictor == 'mf':
		pass
	else:
		pass
		
	run = Run(args, device, predictor)
	run.eval()


def init_log(args, device):
	if not os.path.exists(args.base_log_dir):
		os.makedirs(args.base_log_dir)

	start = datetime.datetime.now()
	logging.basicConfig(level = logging.INFO,
					filename = args.base_log_dir + args.v + '-' + str(time.time()) + '.log',
					filemode = 'a',		# 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
					# a是追加模式，默认如果不写的话，就是追加模式
					)
	logging.info('start! '+str(start))
	logging.info(f'device:{device}')
	logging.info('Parameter:')
	logging.info(str(args))
	logging.info('\n-------------------------------------------------------------\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters for Off Line Evaluation")
	parser.add_argument('--v', default="v")
	parser.add_argument('--choose', default="rl")				# 评估 RL+Predictor 还是 predictor
	parser.add_argument('--topk', type=int, default=10)
	parser.add_argument('--num_thread', type=int, default=4)	# 用 GPU 跑时设为 0

	parser.add_argument('--base_log_dir', default="eval/")
	parser.add_argument('--base_data_dir', default='../data/ml_1M_row/')
	parser.add_argument('--predictor', default='fm')	# fm/ncf
	# embedding
	parser.add_argument('--max_uid', type=int, default=610)		# 1~610
	parser.add_argument('--u_emb_dim', type=int, default=64)
	parser.add_argument('--max_mid', type=int, default=9741)	# 0~9741
	parser.add_argument('--m_emb_dim', type=int, default=128)
	parser.add_argument('--g_emb_dim', type=int, default=32)	# genres emb dim
	# FM
	parser.add_argument('--fm_feature_size', type=int, default=22)	# 还要原来基础加上 actor_output
	parser.add_argument('--k', type=int, default=64)
	# NCF
	parser.add_argument('--n_act', default='elu')
	parser.add_argument('--layers', default='1024,512')
	# 占位
	parser.add_argument("--predictor_lr", type=float, default=0)
	parser.add_argument('--predictor_optim', default='adam')
	parser.add_argument('--momentum', type=float, default=0.8)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--init_std', type=float, default=0.1)

	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tmp = 'cuda' if torch.cuda.is_available() else 'cpu'
	args.num_thread = 0 if tmp == 'cuda' else args.num_thread

	print(f'device:{device}')
	init_log(args, device)
	main(args, device)