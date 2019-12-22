import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '..')		# 使用绝对路径

from PG import PolicyGradient
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle


def test_pg():
	state = env.reset()
	total_reward = 0
	while True:
		# env.render()
		state = state.reshape(1, env.observation_space.shape[0])
		prob, action_i = pg.choose_action_greedy(state, 1)

		next_state, reward, done, info = env.step(action_i[0].item())
		total_reward += reward
		state = next_state

		if done:
			break
	return total_reward



env = gym.make("CartPole-v1")

pg = PolicyGradient(
	n_features = env.observation_space.shape[0], 
	n_actions = env.action_space.n, 
	n_hidden = 32,
	learning_rate = 1e-3,
	reward_decay = 0.99)

# pg = torch.load('models/PG_CartPole.pkl')

EPOCH = 600
total_reward = 0
plot_loss = []
plot_reward = []

for i in range(EPOCH):
	state = env.reset()
	total_reward = 0
	while True:
		# env.render()
		state = state.reshape(1, env.observation_space.shape[0])	# shape -> (1, x)
		action = pg.choose_action(state)
		next_state, reward, done, info = env.step(action)
		total_reward += reward

		pg.store_transition(state, action, reward)
		state = next_state

		if done:
			loss = pg.learn()
			plot_loss.append(loss)
			print('\r{}/{} Loss:{:.4f} total reward:{}'.format(EPOCH, i+1, loss, total_reward), end='')
			# plot_reward.append(test_pg())	# 评估
			break
print('\n\n---train end---\n')

# torch.save(pg, 'models/PG_CartPole.pkl')

# # 保存图片
# plt.xlabel('episode steps')
# plt.ylabel('LOSS')
# plt.plot(plot_loss)
# # plt.savefig('pic/PG_CartPole-loss.png')

# plt.clf()
# plt.xlabel('episode steps')
# plt.ylabel('game steps')
# plt.plot(plot_step)
# plt.savefig('pic/PG_CartPole-step.png')

# plt.xlabel('episode')
# plt.ylabel('Total Reward')
# plt.plot(plot_reward)
# plt.savefig('pic/PG_CartPole-reward.png')
# plt.show()

def test():
	# 玩 10 局 (测试, 用 greedy)
	TEST_EPOCH = 100
	plot_reward = []
	for i in range(TEST_EPOCH):
		state = env.reset()
		total_reward = 0
		while True:
			# env.render()
			state = state.reshape(1, env.observation_space.shape[0])
			prob, action_i = pg.choose_action_greedy(state, 1)

			next_state, reward, done, info = env.step(action_i[0].item())
			total_reward += reward
			state = next_state

			if done:
				print('{}/{} total reward:{}'.format(TEST_EPOCH, i+1, total_reward))
				plot_reward.append(total_reward)
				break

	plt.clf()		
	plt.xlabel('episode')
	plt.ylabel('Total Reward')
	plt.plot(plot_reward)
	plt.show()

# test()


def generate_data():
	capacity_size = 5000		# 多少回合游戏
	replay_buffer = []		# 每个元素(total_reward, [states, actions, rewards]) 是一回合游戏的数据
	for i in range(capacity_size):
		state = env.reset()
		states = []
		actions = []
		rewards = []
		total_reward = 0
		while True:
			state = state.reshape(1, env.observation_space.shape[0])
			prob, action_i = pg.choose_action_greedy(state, 1)

			next_state, reward, done, _ = env.step(action_i[0].item())
			states.append(state.tolist())
			actions.append(action_i[0].item())
			rewards.append(reward)
			total_reward += reward
			state = next_state

			if done:
				print('\r进度:{}/{}'.format(capacity_size, i+1), end='')
				replay_buffer.append((total_reward, states, actions, rewards))
				break

	return replay_buffer


replay_buffer = generate_data()

def save_generate_data(replay_buffer):
	# 保存数据
	with open('CartPole-v1-data.pkl', 'wb') as file:
		pickle.dump(replay_buffer, file)

save_generate_data(replay_buffer)

def load_generate_data():
	with open('CartPole-v1-data.pkl', 'rb') as file:
		result = pickle.load(file)
	return result