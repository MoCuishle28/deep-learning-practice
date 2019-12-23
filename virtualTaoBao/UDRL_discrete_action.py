# Naive implementation of "Training Agents usingUpside-Down Reinforcement Learning"
# Paper link: https://arxiv.org/pdf/1912.02877.pdf
# git clone from https://github.com/parthchadha/upsideDownRL
import numpy as np
import gym
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pdb
from collections import deque
from sortedcontainers import SortedDict
import random
import os, time, sys
import matplotlib.pyplot as plt
import pickle


class BehaviorFunc(nn.Module):
	def __init__(self, state_size, action_size, args):
		super(BehaviorFunc, self).__init__()
		self.args = args
		self.fc1 = nn.Linear(state_size, self.args.hidden_size)
		self.fc2 = nn.Linear(2, self.args.hidden_size)
		self.fc3 = nn.Linear(self.args.hidden_size, action_size)
		self.command_scale = args.command_scale

	def forward(self, state, desired_return, desired_horizon):
		# x = torch.relu(self.fc1(state))
		x = torch.sigmoid(self.fc1(state))	# 效果更好

		concat_command = torch.cat((desired_return, desired_horizon), 1)*self.command_scale

		y = torch.sigmoid(self.fc2(concat_command))		# 论文做法

		x = x * y		# 论文做法
		return self.fc3(x)


class UpsideDownRL(object):
	def __init__(self, env, args):
		super(UpsideDownRL, self).__init__()
		self.env = env
		self.args = args
		self.nb_actions  = self.env.action_space.n
		self.state_space = self.env.observation_space.shape[0]

		# Use sorted dict to store experiences gathered. 
		# This helps in fetching highest reward trajectories during exploratory stage. 
		self.experience	 = SortedDict()
		self.B = BehaviorFunc(self.state_space, self.nb_actions, args)
		self.optimizer = optim.Adam(self.B.parameters(), lr=self.args.lr)
		self.use_random_actions = True # True for the first training epoch.
		self.softmax = nn.Softmax(dim=1)
		# Used to clip rewards so that B does not get unrealistic expected reward inputs.
		self.lunar_lander_max_reward = 500


	# Generate an episode using given command inputs to the B function.
	def gen_episode(self, dr, dh, evaluate=False):
		state = self.env.reset()
		episode_data = []
		states = []
		rewards = []
		actions = []
		total_reward = 0
		while True:
			action = self.select_action(state, dr, dh, evaluate=evaluate)
			next_state, reward, is_terminal, _ = self.env.step(action)
			if self.args.render:
				self.env.render()
			states.append(state)
			actions.append(action)
			rewards.append(reward)
			total_reward += reward
			state = next_state
			dr = min(dr - reward, self.lunar_lander_max_reward)
			dh = max(dh - 1, 1)
			if is_terminal:
				break

		return total_reward, states, actions, rewards


	# Fetch the desired return and horizon from the best trajectories in the current replay buffer
	# to sample more trajectories using the latest behavior function.
	def fill_replay_buffer(self):
		dr, dh = self.get_desired_return_and_horizon()
		self.experience.clear()
		for i in range(self.args.replay_buffer_capacity):
			total_reward, states, actions, rewards = self.gen_episode(dr, dh)
			# total_reward: ( [states...], [actions...], [rewards...])
			self.experience.__setitem__(total_reward, (states, actions, rewards))
			print('\rcapacity:{}/{}'.format(i+1, self.args.replay_buffer_capacity), end='')	# debug
		print()

		if self.args.verbose:
			if self.use_random_actions:
				print("Filled replay buffer with random actions")
			else:
				print("Filled replay buffer using BehaviorFunc")
		self.use_random_actions = False


	def select_action(self, state, desired_return=None, desired_horizon=None, evaluate=False):
		state = np.array(state, dtype=np.float32)
		if self.use_random_actions:
			action = np.random.randint(self.nb_actions)
		else:
			action_prob = self.B(torch.from_numpy(state), 
									torch.from_numpy(np.array(desired_return, dtype=np.float32)).reshape(-1, 1), 
									torch.from_numpy(np.array(desired_horizon, dtype=np.float32).reshape(-1, 1)),
								)
			action_prob = self.softmax(action_prob)

			if not evaluate:
			# # create a categorical distribution over action probabilities
				dist = Categorical(action_prob)
				action = dist.sample().item()	# 相当于按照 softmax 的概率选择
			else:
				# 评估时改成 greedy
				_, action = action_prob.topk(1)
				action = action[0].item()
		return action


	# Todo: don't popitem from the experience buffer since these best-performing trajectories can have huge impact on learning of B
	def get_desired_return_and_horizon(self):
		if (self.use_random_actions):
			return 0, 0

		h = []
		r = []
		for i in range(self.args.explore_buffer_len):
			if len(self.experience) == 0:
				break
			# epsiode 包括了 key 和 value 组成的元组
			episode = self.experience.popitem() # will return in sorted order
			h.append(len(episode[1][0]))
			r.append(episode[0])

		mean_horizon_len = np.mean(h)
		mean_reward = np.random.uniform(low=np.mean(r), high=np.mean(r)+np.std(r))
		return mean_reward, mean_horizon_len


	def trainBehaviorFunc(self, iterations):
		# experience_dict = dict(self.experience)		# 失去了顺序, 但是下面的随机选择不还是无序吗？
		experience_values = list(self.experience.values())
		loss_list = []	
		if len(experience_values) <= 0:
			print('experience_values is empty! [TODO]')
			return loss_list

		for i in range(self.args.train_iter):
			state = []
			dr = []
			dh = []
			target = []
			indices = np.random.choice(len(experience_values), self.args.batch_size, replace=True)
			train_episodes = [experience_values[i] for i in indices]
			# 随机选择 t1
			t1 = [np.random.choice(len(e[0])-2, 1)  for e in train_episodes]

			for pair in zip(t1, train_episodes):
				state.append(pair[1][0][pair[0][0]])
				dr.append(np.sum(pair[1][2][pair[0][0]:]))	# t2 = T - 1
				dh.append(len(pair[1][0])-pair[0][0])	
				target.append(pair[1][1][pair[0][0]])

			self.optimizer.zero_grad()

			state = torch.from_numpy(np.array(state, dtype=np.float32))
			dr = torch.from_numpy(np.array(dr, dtype=np.float32).reshape(-1,1))
			dh = torch.from_numpy(np.array(dh, dtype=np.float32).reshape(-1,1))
			target = torch.from_numpy(np.array(target)).long()
			action_logits = self.B(state, dr, dh)
			loss = nn.CrossEntropyLoss()
			output = loss(action_logits, target).mean()
			print('\r{}/{} LOSS:{:.4f}'.format(i+1, self.args.train_iter, output.item()), end='')
			loss_list.append(output.item())
			# plot_loss(loss_list)

			output.backward()
			self.optimizer.step()
		print()
		# plot_loss(loss_list, fig_name='iterations_'+str(iterations), save=True)
		return loss_list


	# Evaluate the agent using the initial command input from the best topK performing trajectories.
	def evaluate(self):
		testing_rewards = []
		dr, dh = self.get_desired_return_and_horizon()
		for i in range(self.args.evaluate_trials):
			total_reward, states, actions, rewards = self.gen_episode(dr, dh, evaluate=True)	# 玩一轮游戏
			testing_rewards.append(total_reward)

		testing_rewards_mean = np.mean(testing_rewards)
		print("Mean reward achieved : {}".format(testing_rewards_mean))
		return testing_rewards_mean, testing_rewards


	def train(self):
		# Fill replay buffer with random actions for the first time.
		self.fill_replay_buffer()
		iterations = 0
		test_returns = []
		testing_rewards = []
		loss_list = []
		while True:
			# Train behavior function with trajectories stored in the replay buffer.
			tmp_loss_list = self.trainBehaviorFunc(iterations)
			loss_list.extend(tmp_loss_list)
			self.fill_replay_buffer()
			iterations += 1

			# 每轮训练都评估一次
			print('iterations:{}/{}'.format(iterations, self.args.epoch))
			testing_rewards_mean, tmp_list = self.evaluate()
			testing_rewards.extend(tmp_list)
			test_returns.append(testing_rewards_mean)
			
			if iterations == self.args.epoch:
				torch.save(self.B.state_dict(), os.path.join(self.args.save_path, "model.pkl"))
				np.save(os.path.join(self.args.save_path, "testing_rewards_mean"), test_returns)
				np.save(os.path.join(self.args.save_path, "testing_rewards"), testing_rewards)

				# plot_reward(test_returns, fig_name='mean_reward')
				plot_loss(loss_list, fig_name='total_loss', save=True)
				plot_reward(testing_rewards, fig_name='reward')
				break


	def pre_train(self):
		with open('UDRL_data/CartPole-v1-data.pkl', 'rb') as file:
			replay_buffer = pickle.load(file)

		for i, element in enumerate(replay_buffer):
			total_reward, states, actions, rewards = element[0], element[1], element[2], element[3]
			self.experience.__setitem__(total_reward, (states, actions, rewards))
			if (i+1) % self.args.replay_buffer_capacity == 0:
				# train
				experience_values = list(self.experience.values())
				for j in range(self.args.train_iter*2):
					state = []
					dr = []
					dh = []
					target = []
					indices = np.random.choice(len(experience_values), self.args.batch_size, replace=True)
					train_episodes = [experience_values[i] for i in indices]
					t1 = [np.random.choice(len(e[0])-2, 1) if len(e[0])>2 else [0]  for e in train_episodes]

					for pair in zip(t1, train_episodes):
						state.append(pair[1][0][pair[0][0]])
						dr.append(np.sum(pair[1][2][pair[0][0]:]))	# t2 = T - 1
						dh.append(len(pair[1][0])-pair[0][0])	
						target.append(pair[1][1][pair[0][0]])

					self.optimizer.zero_grad()

					state = torch.from_numpy(np.array(state, dtype=np.float32).reshape((len(state), self.env.observation_space.shape[0])))
					dr = torch.from_numpy(np.array(dr, dtype=np.float32).reshape(-1,1))
					dh = torch.from_numpy(np.array(dh, dtype=np.float32).reshape(-1,1))
					target = torch.from_numpy(np.array(target)).long()

					action_logits = self.B(state, dr, dh)
					loss = nn.CrossEntropyLoss()
					output = loss(action_logits, target).mean()
					print('\repisode:{}/{} {}/{} LOSS:{:.8f}'.format(len(replay_buffer), i+1, self.args.train_iter*2, j+1, output.item()), end='')
					output.backward()
					self.optimizer.step()

				print()
				self.evaluate()		# 评估一下
				self.experience.clear()
		print('pretrain end!')


def plot_loss(loss_list, fig_name='demo', save=False):
	plt.ion()
	plt.cla()
	plt.title('UDRL')
	plt.plot(np.array(range(len(loss_list))), loss_list)
	plt.xlabel('episode steps')
	plt.ylabel('LOSS')
	plt.show()
	if save:
		plt.savefig('pic/'+fig_name+".png")
	plt.pause(0.1)


def plot_reward(test_returns, fig_name, save=True):
	plt.cla()
	plt.title('UDRL')
	plt.plot(np.array(range(len(test_returns))), test_returns)
	plt.xlabel('evaluate epoch')
	plt.ylabel('Reward')
	plt.show()
	if save:
		plt.savefig('pic/'+fig_name+".png")


def main():
	# python train.py --render
	parser = argparse.ArgumentParser(description="Hyperparameters for UpsideDown RL")
	parser.add_argument("--render", action='store_true')
	parser.add_argument("--verbose", action='store_true')
	parser.add_argument("--lr", type=float, default=1e-2)
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--hidden_size", type=int, default=64)
	parser.add_argument("--command_scale", type=float, default=0.01)
	parser.add_argument("--replay_buffer_capacity", type=int, default=500)
	parser.add_argument("--explore_buffer_len", type=int, default=10)
	parser.add_argument("--eval_every_k_epoch", type=int, default=3)
	parser.add_argument("--epoch", type=int, default=30)
	parser.add_argument("--evaluate_trials", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=1024)
	parser.add_argument("--train_iter", type=int, default=50)	# 20 epoch 左右可以达到 500
	parser.add_argument("--save_path", type=str, default="UDRL_data/")
	
	args = parser.parse_args()
	if not os.path.exists(args.save_path):
		os.mkdir(args.save_path)

	# 替代一下先
	env = gym.make("CartPole-v1")

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	agent = UpsideDownRL(env, args)
	agent.pre_train()	# 效果更差... 什么问题？
	agent.train()
	env.close()


if __name__ == "__main__":
	main()