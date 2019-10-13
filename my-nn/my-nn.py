import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import pylab


class neuralNetwork:
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate

		# input -> hidden 的权重矩阵 初始胡为 -0.5 ~ 0.5
		# self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
		# self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

		# 正态分布的随机初始化
		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

		self.activation_function = lambda x: scipy.special.expit(x)


	def train(self, inputs_list, targets_list):
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		# 计算误差
		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)

		self.who += self.lr * np.dot((output_errors*final_outputs*(1.0 - final_outputs)), np.transpose(hidden_outputs))
		self.wih += self.lr * np.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)), np.transpose(inputs))

	# 接受输入，返回神经网络的输出
	def query(self, inputs_list):
		# 转换 list 为 2D 矩阵
		inputs = np.array(inputs_list, ndmin=2).T

		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		return final_outputs


	def __str__(self):
		return str(self.inodes)+","+str(self.hnodes)+","+str(self.onodes)+","+str(self.lr)



if __name__ == '__main__':
	input_nodes = 784
	hidden_nodes = 200
	output_nodes = 10
	learning_rate = 0.1

	n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

	training_data_file = open('mnist_dataset/mnist_train.csv', 'r')
	training_data_list = training_data_file.readlines()
	training_data_file.close()

	epochs = 5

	for e in range(epochs):
		for record in training_data_list:
			all_values = record.split(',')
			inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			targets = np.zeros(output_nodes) + 0.01
			targets[int(all_values[0])] = 0.99
			n.train(inputs, targets)

	test_data_file = open('mnist_dataset/mnist_test.csv', 'r')
	test_data_list = test_data_file.readlines()
	test_data_file.close()

	scorecard = []
	for record in test_data_list:
		all_values = record.split(',')
		correct_label = int(all_values[0])
		inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		outputs = n.query(inputs)
		label = np.argmax(outputs)	# 返回索引
		print('correct label:', correct_label, " network's answer:", label)
		if label == correct_label:
			scorecard.append(1)
		else:
			scorecard.append(0)

	scorecard_array = np.array(scorecard)
	print('performance = ', scorecard_array.sum() / scorecard_array.size)


	# 看一下图片
	# data_file = open('mnist_dataset/mnist_train_100.csv', 'r')
	# data_list = data_file.readlines()
	# data_file.close()
	# all_values = data_list[0].split(',')
	# image_array = np.asfarray(all_values[1:]).reshape((28, 28))		# 将文本字符串转为实数, 并转为28x28矩阵
	# plt.imshow(image_array, cmap='Greys', interpolation='None')
	# pylab.show()