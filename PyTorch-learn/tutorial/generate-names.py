from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import random
import time
import math

import torch
import torch.nn as nn


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path):
	return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)


# Read a file and split into lines
def readLines(filename):
	lines = open(filename, encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line) for line in lines]


# Build the category_lines dictionary, a list of lines per category
category_lines = {}		# state:[prename 1, prename 2, ...]
all_categories = []		# state 1, state 2, ...  18 个国家的语言
for filename in findFiles('data/NLP-tutorial-0/names/*.txt'):
	category = os.path.splitext(os.path.basename(filename))[0]
	all_categories.append(category)
	lines = readLines(filename)
	category_lines[category] = lines

n_categories = len(all_categories)


class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size

		self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
		self.o2o = nn.Linear(hidden_size + output_size, output_size)
		self.dropout = nn.Dropout(0.1)
		self.softmax = nn.LogSoftmax(dim=1)


	def forward(self, category, input_data, hidden):
		input_combined = torch.cat((category, input_data, hidden), 1)
		hidden = self.i2h(input_combined)
		output = self.i2o(input_combined)
		output_combined = torch.cat((hidden, output), 1)
		output = self.o2o(output_combined)
		output = self.dropout(output)
		output = self.softmax(output)
		return output, hidden


	def initHidden(self):
		return torch.zeros(1, self.hidden_size)


# Random item from a list
def randomChoice(l):
	return l[random.randint(0, len(l) - 1)]


# Get a random category and random line from that category
def randomTrainingPair():
	category = randomChoice(all_categories)
	line = randomChoice(category_lines[category])	# 在该国家中随机选一个名
	return category, line


# One-hot vector for category
def categoryTensor(category):
	li = all_categories.index(category)
	tensor = torch.zeros(1, n_categories)
	tensor[0][li] = 1
	return tensor


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
	tensor = torch.zeros(len(line), 1, n_letters)
	for li in range(len(line)):
		letter = line[li]
		tensor[li][0][all_letters.find(letter)] = 1
	return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
	letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
	letter_indexes.append(n_letters - 1) # EOS
	return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
	category, line = randomTrainingPair()
	category_tensor = categoryTensor(category)
	input_line_tensor = inputTensor(line)
	target_line_tensor = targetTensor(line)
	return category_tensor, input_line_tensor, target_line_tensor

# target 就是 input 的后一个字母; category 是 one-hot 代表是那个国家
# category, input_data, target = randomTrainingExample()
# print(all_categories[category.topk(1)[-1]])
# print(input_data.shape)
# print(all_letters[input_data[0].topk(1)[-1]], all_letters[target[0]])
# print(all_letters[input_data[1].topk(1)[-1]], all_letters[target[0]])
# print(all_letters[input_data[2].topk(1)[-1]], all_letters[target[2]])


########## training ##########

criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
	target_line_tensor.unsqueeze_(-1)
	hidden = rnn.initHidden()

	rnn.zero_grad()

	loss = 0

	for i in range(input_line_tensor.size(0)):
		output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
		l = criterion(output, target_line_tensor[i])
		loss += l

	loss.backward()

	for p in rnn.parameters():
		p.data.add_(-learning_rate, p.grad.data)

	return output, loss.item() / input_line_tensor.size(0)


def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


rnn = RNN(n_letters, 128, n_letters)

n_iters = 10000
print_every = 500
plot_every = 50
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
	output, loss = train(*randomTrainingExample())
	total_loss += loss

	if iter % print_every == 0:
		print('%s (%d %d%%) Loss:%.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

	if iter % plot_every == 0:
		all_losses.append(total_loss / plot_every)
		total_loss = 0


# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# plt.figure()
# plt.plot(all_losses)
# plt.show()


max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


samples('Russian', 'RUS')	# 每个字母生成一个最长为 max_length 的名字

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')