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
import virtualTB
import io


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
		x = torch.sigmoid(self.fc1(state))

		concat_command = torch.cat((desired_return, desired_horizon), 1)*self.command_scale

		y = torch.sigmoid(self.fc2(concat_command))		# 论文做法

		x = x * y		# 论文做法
		return self.fc3(x)


class UpsideDownRL(object):
	def __init__(self, env, args):
		super(UpsideDownRL, self).__init__()
		self.env = env
		self.args = args
		# self.nb_actions  = self.env.action_space.n
		self.nb_actions = env.action_space.shape[0]
		self.state_space = self.env.observation_space.shape[0]

		# Use sorted dict to store experiences gathered. 
		# This helps in fetching highest reward trajectories during exploratory stage. 
		self.experience	 = SortedDict()
		self.B = BehaviorFunc(self.state_space, self.nb_actions, args)
		self.optimizer = optim.Adam(self.B.parameters(), lr=self.args.lr)
		self.use_random_actions = True # True for the first training epoch.
		# Used to clip rewards so that B does not get unrealistic expected reward inputs.
		self.virtual_taobao_max_reward = 100


	# Generate an episode using given command inputs to the B function.
	def gen_episode(self, dr, dh):
		state = self.env.reset()
		episode_data = []
		states = []
		rewards = []
		actions = []
		total_reward = 0
		while True:
			action = self.select_action(state, dr, dh)
			action = action.reshape(27,)
			next_state, reward, is_terminal, _ = self.env.step(action)
			if self.args.render:
				self.env.render()
			states.append(state)
			actions.append(action)
			rewards.append(reward)
			total_reward += reward
			state = next_state
			dr = min(dr - reward, self.virtual_taobao_max_reward)
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
			print('\rcapacity:{}/{}'.format(i+1, self.args.replay_buffer_capacity), end='')
		print()

		if self.args.verbose:
			if self.use_random_actions:
				print("Filled replay buffer with random actions")
			else:
				print("Filled replay buffer using BehaviorFunc")
		self.use_random_actions = False


	def select_action(self, state, desired_return=None, desired_horizon=None):
		state = np.array(state, dtype=np.float32)
		if self.use_random_actions:
			action = self.env.action_space.sample()
		else:
			action = self.B(torch.from_numpy(state), 
									torch.from_numpy(np.array(desired_return, dtype=np.float32)).reshape(-1, 1), 
									torch.from_numpy(np.array(desired_horizon, dtype=np.float32).reshape(-1, 1)),
								).detach().numpy()
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
		for i in range(self.args.train_iter):
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

			state = torch.from_numpy(np.array(state, dtype=np.float32))
			# state = torch.tensor(state, dtype=torch.float32)

			dr = torch.from_numpy(np.array(dr, dtype=np.float32).reshape(-1,1))
			dh = torch.from_numpy(np.array(dh, dtype=np.float32).reshape(-1,1))

			# target = torch.from_numpy(np.array(target, dtype=np.float32))
			target = torch.tensor(target, dtype=torch.float32)

			action = self.B(state, dr, dh)
			loss = nn.MSELoss()
			output = loss(action, target)
			print('\r{}/{} LOSS:{:.8f}'.format(i+1, self.args.train_iter, output.item()), end='')
			loss_list.append(output.item())
			# plot_loss(loss_list)

			output.backward()
			self.optimizer.step()
		print()
		# plot_loss(loss_list, fig_name='iterations_'+str(iterations), save=True)
		return loss_list


	# Evaluate the agent using the initial command input from the best topK performing trajectories.
	def evaluate(self):
		ctr_list = []
		dr, dh = self.get_desired_return_and_horizon()
		for i in range(self.args.evaluate_trials):
			total_reward, states, actions, rewards = self.gen_episode(dr, dh)	# 玩一轮游戏
			ctr_list.append( total_reward*1.0 / (len(states) * 10.0) )

		testing_CTR_mean = np.mean(ctr_list)
		print("Mean CTR achieved : {:.6f}%".format(testing_CTR_mean*100))
		return testing_CTR_mean, ctr_list


	def train(self):
		# Fill replay buffer with random actions for the first time.
		self.evaluate()
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
				torch.save(self.B.state_dict(), os.path.join(self.args.save_path, "continue_model.pkl"))
				np.save(os.path.join(self.args.save_path, "continue_testing_rewards_mean"), test_returns)
				np.save(os.path.join(self.args.save_path, "continue_testing_rewards"), testing_rewards)

				# plot_reward(test_returns, fig_name='mean_reward')
				plot_loss(loss_list, fig_name='total_loss', save=True)
				plot_reward(testing_rewards, fig_name='reward')
				break


	def pre_train(self):
		features, labels, clicks = load_data()
		states = []
		rewards = []
		actions = []
		total_reward = 0
		cnt0 = 0
		cnt1 = 0
		iterations = 0
		for state, action, reward in zip(features, labels, clicks):
			states.append(np.array(state))
			actions.append(np.array(action))
			rewards.append(np.array(reward))
			total_reward += reward
			cnt0 += 1
			if cnt0 == 16:
				cnt0 = 0
				self.experience.__setitem__(total_reward, (states, actions, rewards))
				iterations += 1
				cnt1 += 1
				if cnt1 == self.args.replay_buffer_capacity:
					cnt1 = 0

					############ train ############
					experience_values = list(self.experience.values())
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

					state = torch.from_numpy(np.array(state, dtype=np.float32))
					dr = torch.from_numpy(np.array(dr, dtype=np.float32).reshape(-1,1))
					dh = torch.from_numpy(np.array(dh, dtype=np.float32).reshape(-1,1))
					target = torch.tensor(target, dtype=torch.float32)

					action = self.B(state, dr, dh)
					loss = nn.MSELoss()
					output = loss(action, target)
					print('{}/{} LOSS:{:.8f}'.format(iterations, 100000//16, output.item()))

					output.backward()
					self.optimizer.step()

					self.experience.clear()
		self.experience.clear()


	def test(self):
		pass


def load_data():
	features, labels, clicks = [], [], []
	with io.open('data/dataset.txt','r') as file:
		for line in file:
			features_l, labels_l, clicks_l = line.split('\t')
			features.append([float(x) for x in features_l.split(',')])
			labels.append([float(x) for x in labels_l.split(',')])
			clicks.append(int(clicks_l))
	return features, labels, clicks


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
	plt.ylabel('CTR')
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
	parser.add_argument("--epoch", type=int, default=10)
	parser.add_argument("--evaluate_trials", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=1024)
	parser.add_argument("--train_iter", type=int, default=100)
	parser.add_argument("--save_path", type=str, default="UDRL_data/")
	
	args = parser.parse_args()
	if not os.path.exists(args.save_path):
		os.mkdir(args.save_path)

	# 替代一下先
	env = gym.make("VirtualTB-v0")

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	agent = UpsideDownRL(env, args)
	agent.pre_train()
	agent.train()
	env.close()


if __name__ == "__main__":
	main()