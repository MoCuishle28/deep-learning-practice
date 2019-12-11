import numpy as np
import tensorlayer as tl
import tensorflow as tf


rewards = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
gamma = 0.9
discount_rewards = tl.rein.discount_episode_rewards(rewards, gamma)
print(discount_rewards)

discount_rewards = tl.rein.discount_episode_rewards(rewards, gamma, mode=1)
print(discount_rewards)



# example：
# states_batch_pl = tf.placeholder(tf.float32, shape=[None, D])
# network = InputLayer(states_batch_pl, name='input')
# network = DenseLayer(network, n_units=H, act=tf.nn.relu, name='relu1')
# network = DenseLayer(network, n_units=3, name='out')
# probs = network.outputs
# sampling_prob = tf.nn.softmax(probs)
# actions_batch_pl = tf.placeholder(tf.int32, shape=[None])
# discount_rewards_batch_pl = tf.placeholder(tf.float32, shape=[None])
# loss = tl.rein.cross_entropy_reward_loss(probs, actions_batch_pl, discount_rewards_batch_pl)
# train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)


prob = tf.constant([0.2, 0.4, 0.4])
# prob = np.array([0.2, 0.4, 0.4])
for _ in range(5):
	a = tl.rein.choice_action_by_probs(prob)
	# 1 <class 'numpy.int32'> [1] <class 'numpy.ndarray'>
	print(a, type(a), a.ravel(), type(a.ravel()))


print('----')
for _ in range(5):
	a = tl.rein.choice_action_by_probs([0.2, 0.4, 0.4], ['a', 'b', 'c'])
	print(a, type(a))	# c <class 'numpy.str_'>