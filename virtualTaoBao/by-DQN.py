import tensorflow as tf
import numpy as np
import gym
import virtualTB
import random
from collections import deque


num_episodes = 500              # 游戏训练的总episode数量
num_exploration_episodes = 100  # 探索过程所占的episode数量
max_len_episode = 1000          # 每个episode的最大回合数
batch_size = 32                 # 批次大小
learning_rate = 1e-3            # 学习率
gamma = 1.                      # 折扣因子
initial_epsilon = 1.            # 探索起始时的探索率
final_epsilon = 0.01            # 探索终止时的探索率

'''
TODO
如何用上 Q-learning ???
相当于推荐动作 action 是一个连续的动作
'''

class QNetwork(tf.keras.Model):
	def __init__(self, input_size, output_size):
		super().__init__()
		# 输入必须是一个 batch 的矩阵吗？
		self.dense1 = tf.keras.layers.Dense(units=input_size, activation=tf.nn.relu)
		self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
		self.dense3 = tf.keras.layers.Dense(units=output_size)


	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
		return x


	def predict(self, inputs):
		q_values = self(inputs)
		return tf.argmax(q_values, axis=-1)


if __name__ == '__main__':
	env = gym.make('VirtualTB-v0')       # 实例化一个游戏环境，参数为游戏名称
	model = QNetwork(env.observation_space.shape[0], env.action_space.shape[0])

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	replay_buffer = deque(maxlen=10000) # 使用一个 deque 作为 Q Learning 的经验回放池
	epsilon = initial_epsilon

	for episode_id in range(num_episodes):
		state = env.reset()             # 初始化环境，获得初始状态

		epsilon = max(                  # 计算当前探索率
			initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
			final_epsilon)

		cnt = 0
		ct = 0
		for t in range(max_len_episode):
			if random.random() < epsilon:               # epsilon-greedy 探索策略，以 epsilon 的概率选择随机动作
				action = env.action_space.sample()      # 选择随机动作（探索）
			else:
				# 选择模型计算出的 Q Value 最大的动作
				# 为什么要 expand_dims() 加一维到 0
				# 因为输入必须是一个 batch 的矩阵，这里只有一条example输入，所以扩展一维相当于这个 batch 只有一条数据(shape 从 (4,) 到 (1, 4))
				action = model.predict(np.expand_dims(state, axis=0)).numpy()
				action = action[0]

			# 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
			next_state, reward, done, info = env.step(action)
			cnt += 1
			ct += reward

			# 将(state, action, reward, next_state)的四元组（外加 done 标签表示是否结束）放入经验回放池
			replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
			# 更新当前 state
			state = next_state

			if done:                                    # 游戏结束则退出本轮循环，进行下一个 episode
				print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
				break

			if len(replay_buffer) >= batch_size:
				# 从经验回放池中随机取一个批次的四元组 
				batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
					*random.sample(replay_buffer, batch_size))
				# 并分别转换为 NumPy 数组
				batch_state, batch_reward, batch_next_state, batch_done = \
					[np.array(a, dtype=np.float64) for a in [batch_state, batch_reward, batch_next_state, batch_done]]

				batch_action = np.array(batch_action, dtype=np.int32)

				q_value = model(batch_next_state)
				y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)  # 计算 y 值
				
				with tf.GradientTape() as tape:
					loss = tf.keras.losses.mean_squared_error(  # 最小化 y 和 Q-value 的距离
						y_true=y,
						y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=env.action_space.shape[0]), axis=1)
					)
				grads = tape.gradient(loss, model.variables)
				optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))       # 计算梯度并更新参数

		print('CTR:', ct/cnt, 'cnt:', cnt)