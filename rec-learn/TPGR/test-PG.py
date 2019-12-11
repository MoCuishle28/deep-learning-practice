import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '..')		# 使用绝对路径

from PG import PolicyGradient
import torch
import numpy as np
import gym


env = gym.make("CartPole-v1")

pg = PolicyGradient(
	n_features = env.observation_space.shape[0], 
	n_actions = env.action_space.n, 
	n_hidden = 16,
	learning_rate = 1e-3,
	reward_decay = 0.99)

# pg = torch.load('models/PG_CartPole.pkl')

EPOCH = 1000
plot_loss = []
plot_step = []

for i in range(EPOCH):
	state = env.reset()
	step_cnt = 0
	while True:
		env.render()
		state = state.reshape(1, env.observation_space.shape[0])	# shape -> (1, x)
		action = pg.choose_action(state)
		next_state, reward, done, info = env.step(action)
		step_cnt += 1

		pg.store_transition(state, action, reward)
		state = next_state

		if done:
			loss = pg.learn()
			plot_loss.append(loss)
			plot_step.append(step_cnt)
			print('{}/{} Loss:{:.4f} step:{}'.format(EPOCH, i+1, loss, step_cnt))
			break

torch.save(pg, 'models/PG_CartPole.pkl')

# 保存图片
import matplotlib.pyplot as plt

plt.xlabel('episode steps')
plt.ylabel('LOSS')
plt.plot(plot_loss)
plt.savefig('pic/PG_CartPole-loss.png')

plt.clf()
plt.xlabel('episode steps')
plt.ylabel('game steps')
plt.plot(plot_step)
plt.savefig('pic/PG_CartPole-step.png')
plt.show()


# 玩 10 局
TEST_EPOCH = 10
for i in range(TEST_EPOCH):
	state = env.reset()
	step_cnt = 0
	total_reward = 0
	while True:
		env.render()
		state = state.reshape(1, env.observation_space.shape[0])
		prob, action_i = pg.choose_action_greedy(state, 1)

		next_state, reward, done, info = env.step(action_i[0].item())
		step_cnt += 1
		total_reward += reward
		state = next_state

		if done:
			print('{}/{} Step:{}'.format(TEST_EPOCH, i+1, step_cnt))
			break