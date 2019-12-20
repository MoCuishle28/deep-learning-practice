import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '..')		# 使用绝对路径

from PG import PolicyGradient
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt


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

EPOCH = 500
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
			plot_reward.append(test_pg())
			break

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

plt.xlabel('episode')
plt.ylabel('Total Reward')
plt.plot(plot_reward)
plt.savefig('pic/PG_CartPole-reward.png')
# plt.show()


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