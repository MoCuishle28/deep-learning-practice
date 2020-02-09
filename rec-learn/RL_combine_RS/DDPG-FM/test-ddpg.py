import gym
import math
import torch
import random
import time, sys
import numpy as np

from ddpg import DDPG
from ddpg import Transition
from ddpg import ReplayMemory
from ddpg import OUNoise

import matplotlib.pyplot as plt
import logging
import datetime
import time


env = gym.make('Pendulum-v0')

env.seed(0)
np.random.seed(0)
torch.manual_seed(0)

actor_path = 'models/actor-v0'
critic_path = 'models/critic-v0'

log_name = time.time()

logging.basicConfig(level = logging.INFO,
					filename = "../data/ddpg-fm/test-ddpg/" + str(log_name) + '.log',
					filemode = 'a')

log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logging.info("start! "+str(log_time))

GAMMA = 0.95
tau = 0.01
hidden_size = 64
actor_lr = 1e-3
critic_lr = 1e-3


params = ', '.join(["gamma:"+str(GAMMA), 'tau:'+str(tau), 'hidden_size:'+str(hidden_size), 'actor_lr'+str(actor_lr), 'critic_lr'+str(critic_lr)])
logging.info("params: "+ params)

agent = DDPG(gamma = GAMMA, tau = tau, hidden_size = hidden_size,
					num_inputs = env.observation_space.shape[0], action_space = env.action_space,
					actor_lr=actor_lr, critic_lr=critic_lr)
# 加载模型
# agent.load_model(actor_path, critic_path)

memory = ReplayMemory(5120)

ounoise = OUNoise(env.action_space.shape[0])
param_noise = None

rewards = []
total_numsteps = 0

EPISODE = 100
# train_times = 5
BATCH_SIZE = 512
TEST_EPISODE = 50

train_average_reward_list = []
test_average_reward_list = []
value_loss_list = []
poicy_loss_list = []

for i_episode in range(EPISODE):
	state = torch.Tensor([env.reset()])
	episode_reward = 0
	episode_step = 0
	total_value_loss = 0
	total_poicy_loss = 0
	episode_updates = 0

	while True:
		# env.render()
		action = agent.select_action(state, ounoise, param_noise)	# 加入噪声, exploration
		# action = agent.select_action(state)	# 不加噪声

		next_state, reward, done, _ = env.step(action.numpy()[0])
		total_numsteps += 1
		episode_step += 1
		episode_reward += reward

		action = torch.Tensor(action)
		mask = torch.Tensor([not done])
		next_state = torch.Tensor([next_state])
		reward = torch.Tensor([reward])

		memory.push(state, action, mask, next_state, reward)

		state = next_state

		if len(memory) >= BATCH_SIZE:
			transitions = memory.sample(BATCH_SIZE)
			batch = Transition(*zip(*transitions))

			# value_loss(先下降后上升, 再下降), policy_loss(上升, 后略有下降) ？ TODO
			value_loss, policy_loss = agent.update_parameters(batch)
			value_loss_list.append(value_loss)
			poicy_loss_list.append(policy_loss)
			total_value_loss += value_loss
			total_poicy_loss += policy_loss
			episode_updates += 1

		if done:
			if episode_updates == 0:
				episode_updates = 1

			log_info = ("{}/{}, step:{}, total reward:{:.4f}, average reward:{:.6f}," + 
						" mean value loss:{:.4f}, mean policy loss:{:.4f}").format(
				i_episode, EPISODE, episode_step, episode_reward, 
				episode_reward / episode_step, total_value_loss/episode_updates, total_poicy_loss/episode_updates)

			print(log_info)
			log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			logging.info(str(log_time)+" -> "+log_info)
			train_average_reward_list.append(episode_reward / episode_step)
			break

	if i_episode % 5 == 0:
		episode_reward = 0
		episode_step = 0
		for i in range(TEST_EPISODE):
			state = torch.Tensor([env.reset()])
			while True:
				action = agent.select_action(state)		# 不加噪声
				
				next_state, reward, done, info = env.step(action.numpy()[0])
				episode_reward += reward
				episode_step += 1

				next_state = torch.Tensor([next_state])

				state = next_state
				if done:
					break

		# rewards.append(episode_reward)
		log_info = "TEST Episode: {}, episode_step: {}, average reward: {:.6f}".format(i_episode, episode_step, episode_reward / episode_step)
		print(log_info)
		log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		logging.info(str(log_time)+" -> "+log_info)
		test_average_reward_list.append(episode_reward / episode_step)

env.close()

plt.subplot(1, 5, 1)
plt.title('Train Average Reward')
plt.xlabel('episode')
plt.ylabel('Reward')
plt.plot([i for i in range(len(train_average_reward_list))], train_average_reward_list)

plt.subplot(1, 5, 3)
plt.title('Test Average Reward')
plt.xlabel('Test Times')
plt.ylabel('Reward')
plt.plot([i for i in range(len(test_average_reward_list))], test_average_reward_list)

plt.subplot(1, 5, 5)
plt.title('LOSS')
plt.xlabel('update times')
plt.ylabel('LOSS')
value_line, = plt.plot([i for i in range(len(value_loss_list))], value_loss_list, label='Value Loss', color='blue')
policy_line, = plt.plot([i for i in range(len(poicy_loss_list))], poicy_loss_list, label='Policy Loss', color='red')
plt.legend(handles=[value_line, policy_line], loc='best')

plt.show()

# agent.save_model('taobao', actor_path=actor_path, critic_path=critic_path)