import gym
import math
import argparse
import random
import virtualTB
import time, sys
import configparser
import numpy as np
from gym import wrappers
from copy import deepcopy
import matplotlib.pyplot as plt
import datetime
import io
from DDPG_TL import DDPG
from DDPG_TL import MEMORY_CAPACITY


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')
args = parser.parse_args()


def load_data():
	features, labels, clicks = [], [], []
	with io.open('data/dataset.txt','r') as file:
		for line in file:
			features_l, labels_l, clicks_l = line.split('\t')
			features.append([float(x) for x in features_l.split(',')])
			labels.append([float(x) for x in labels_l.split(',')])
			clicks.append(int(clicks_l))
	return features, labels, clicks


env = gym.make('VirtualTB-v0')

env.seed(0)
np.random.seed(0)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

start = datetime.datetime.now()

EPISODE = 1		# 22 分钟左右一个 epoch
TEST_EPISODE = 100
session_num = 60

reward_buffer = []

if args.train:
	features, labels, clicks = load_data()
	for episode_i in range(EPISODE):
		for i in range(len(features)):
			state = np.array(features[i])
			action = np.array(labels[i])
			reward = np.array(clicks[i])

			if i+1 >= len(features):
				break
			next_state = np.array(features[i+1])

			ddpg.store_transition(state, action, reward, next_state)
			if ddpg.pointer > MEMORY_CAPACITY:
				ddpg.learn()
			print('\r进度[{}|{}] time:{}'.format(len(features), i, datetime.datetime.now()-start), end='')

		if ddpg.pointer > MEMORY_CAPACITY:
			ddpg.learn()
		ddpg.clear_memory()
	ddpg.save_ckpt()


ddpg.load_ckpt()
reward_buffer = []
aver_ctr = []
total_cnt = 0
for i in range(TEST_EPISODE):
	total_reward = 0
	cnt = 0
	s = env.reset()
	# 一回合 session_num 个 session
	for session in range(session_num):
		done = False
		while not done:
			a = ddpg.choose_action(s).numpy()
			_s, r, done, info = env.step(a)
			ddpg.store_transition(s, a, r, _s)
			s = _s
			total_reward += r
			cnt += 1
			if ddpg.pointer > MEMORY_CAPACITY:
				ddpg.learn()
				# ddpg.clear_memory()
	total_cnt += cnt
	print('{}/{} Total Cnt:{} CTR:{:.4f}%'.format(i+1, TEST_EPISODE, total_cnt, (total_reward / cnt / 10)*100))
	aver_ctr.append((total_reward / cnt / 10)*100)
	reward_buffer.append(total_reward / cnt / 10)

	plt.ion()
	plt.cla()
	plt.title('DDPG')
	plt.plot(np.array(range(len(reward_buffer))), reward_buffer)  # plot the episode vt
	plt.xlabel('episode steps')
	plt.ylabel('CTR')
	plt.show()
	plt.pause(0.1)

print('Average CTR:{:.4f}%'.format(sum(aver_ctr)/len(aver_ctr)))
# ddpg.save_ckpt()