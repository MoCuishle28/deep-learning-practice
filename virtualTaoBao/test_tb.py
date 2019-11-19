import gym
import virtualTB

env = gym.make('VirtualTB-v0')

print('env.action_space:', env.action_space)	# Box(27, )
print('env.observation_space:', env.observation_space)	# Box(91, )
print('env.observation_space.low:', env.observation_space.low)
print('env.observation_space.high:', env.observation_space.high)

state = env.reset()
print('-----start recommender---')
while True:
	env.render()
	action = env.action_space.sample()
	print('action:', action)

	state, reward, done, info = env.step(action)

	print('state:', state)
	print('reward:', reward)
	print('info:', info)
	print('---END Episode---')

	if done: 
		break

env.render()	# 会输出一行数据