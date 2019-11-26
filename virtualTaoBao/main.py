import gym
import math
import torch
import random
import virtualTB
import time, sys
import configparser
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import wrappers
from ddpg import DDPG
from copy import deepcopy
from collections import namedtuple
# import matplotlib.pyplot as plt
import logging
import datetime


FLOAT = torch.FloatTensor
LONG = torch.LongTensor

Transition = namedtuple(
	'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))
class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class OUNoise:

	def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
		self.action_dimension = action_dimension
		self.scale = scale
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.state = np.ones(self.action_dimension) * self.mu
		self.reset()

	def reset(self):
		self.state = np.ones(self.action_dimension) * self.mu

	def noise(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
		self.state = x + dx
		return self.state * self.scale

env = gym.make('VirtualTB-v0')

env.seed(0)
np.random.seed(0)
torch.manual_seed(0)

actor_path = 'models/actor-v2'
critic_path = 'models/critic-v2'
log_name = actor_path.split('/')[-1].split('-')[-1]

logging.basicConfig(level = logging.INFO,
					filename = "log/" + log_name + '.log',
					filemode = 'a')

log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logging.info("start! "+str(log_time))

GAMMA = 0.95
tau = 0.001
hidden_size = 128



params = ', '.join(["gamma:"+str(GAMMA), 'tau:'+str(tau), 'hidden_size:'+str(hidden_size)])
logging.info("params: "+ params)

agent = DDPG(gamma = GAMMA, tau = tau, hidden_size = hidden_size,
					num_inputs = env.observation_space.shape[0], action_space = env.action_space)
# 加载模型
# agent.load_model(actor_path, critic_path)

memory = ReplayMemory(1000000)

ounoise = OUNoise(env.action_space.shape[0])
param_noise = None

rewards = []
total_numsteps = 0
updates = 0

EPISODE = 500
BATCH_SIZE = 256

for i_episode in range(EPISODE):
	state = torch.Tensor([env.reset()])
	episode_reward = 0

	while True:
		action = agent.select_action(state, ounoise, param_noise)
		next_state, reward, done, _ = env.step(action.numpy()[0])
		total_numsteps += 1
		episode_reward += reward

		action = torch.Tensor(action)
		mask = torch.Tensor([not done])
		next_state = torch.Tensor([next_state])
		reward = torch.Tensor([reward])

		memory.push(state, action, mask, next_state, reward)

		state = next_state

		if len(memory) >= BATCH_SIZE:
			for _ in range(5):
				transitions = memory.sample(BATCH_SIZE)
				batch = Transition(*zip(*transitions))

				value_loss, policy_loss = agent.update_parameters(batch)

				updates += 1

				if len(memory) < BATCH_SIZE:
					break

				# log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
				# logging.info(str(log_time)+" Loss -> "+str(value_loss)+", "+str(policy_loss))

		if done:
			break

	# rewards.append(episode_reward)
	if i_episode % 10 == 0:
		episode_reward = 0
		episode_step = 0
		for i in range(50):
			state = torch.Tensor([env.reset()])
			while True:
				action = agent.select_action(state)
				
				next_state, reward, done, info = env.step(action.numpy()[0])
				episode_reward += reward
				episode_step += 1

				next_state = torch.Tensor([next_state])

				state = next_state
				if done:
					break

		# rewards.append(episode_reward)
		log_info = "Episode: {}, total numsteps: {}, average reward: {}, CTR: {}".format(i_episode, episode_step, episode_reward / 50, episode_reward / episode_step / 10)
		print(log_info)
		log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		logging.info(str(log_time)+" -> "+log_info)

env.close()

# agent.save_model('taobao', actor_path=actor_path, critic_path=critic_path)