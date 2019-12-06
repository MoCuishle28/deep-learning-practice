from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import random
import time
import math

import unicodedata
import string

import torch
import torch.nn as nn


def findFiles(path):
	return glob.glob(path)

# print(findFiles('data/NLP-tutorial-0/names/*.txt'))


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)

# print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}		# state:[prename 1, prename 2, ...]
all_categories = []		# state 1, state 2, ...  18 个国家的语言

# Read a file and split into lines
def readLines(filename):
	lines = open(filename, encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/NLP-tutorial-0/names/*.txt'):
	category = os.path.splitext(os.path.basename(filename))[0]
	all_categories.append(category)
	lines = readLines(filename)
	category_lines[category] = lines


##### Turning Names into Tensors #####

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
	return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
	tensor = torch.zeros(1, n_letters)
	tensor[0][letterToIndex(letter)] = 1
	return tensor


# Turn a line into a <line_length x 1 x n_letters>, or an array of one-hot letter vectors
def lineToTensor(line):
	tensor = torch.zeros(len(line), 1, n_letters)
	for li, letter in enumerate(line):
		tensor[li][0][letterToIndex(letter)] = 1
	return tensor


# print(letterToTensor('J').size())		# (1, 57)
# print(lineToTensor('Jones').size())	# (5, 1, 57)
# print(lineToTensor('Jones'))


class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		self.hidden_size = hidden_size

		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)


	def forward(self, input_data, hidden):
		combined = torch.cat((input_data, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.softmax(output)
		return output, hidden


	def initHidden(self):
		return torch.zeros(1, self.hidden_size)


n_categories = len(all_categories)
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)
print(output)


def categoryFromOutput(output):
	top_n, top_i = output.topk(1)
	category_i = top_i[0].item()
	return all_categories[category_i], category_i

# print(categoryFromOutput(output))


def randomChoice(l):
	return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
	category = randomChoice(all_categories)			# 选一个国家
	line = randomChoice(category_lines[category])	# 在该国家中选一个姓
	category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
	line_tensor = lineToTensor(line)
	return category, line, category_tensor, line_tensor

# 测试随机选取训练样本
# for i in range(10):
# 	category, line, category_tensor, line_tensor = randomTrainingExample()
# 	print('category =', category, '/ line =', line)


######## train ########
criterion = nn.NLLLoss()

learning_rate = 0.005

def train(category_tensor, line_tensor):
	hidden = rnn.initHidden()

	rnn.zero_grad()

	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)

	loss = criterion(output, category_tensor)
	loss.backward()

	# Add parameters' gradients to their values, multiplied by learning rate
	for p in rnn.parameters():
		p.data.add_(-learning_rate, p.grad.data)

	return output, loss.item()


n_iters = 10000
print_every = 500
plot_every = 100


# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


start = time.time()

for iter in range(1, n_iters + 1):
	category, line, category_tensor, line_tensor = randomTrainingExample()
	output, loss = train(category_tensor, line_tensor)
	current_loss += loss

	# Print iter number, loss, name and guess
	if iter % print_every == 0:
		guess, guess_i = categoryFromOutput(output)
		correct = 'yes' if guess == category else 'no (%s)' % category
		print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

	# Add current loss avg to list of losses
	if iter % plot_every == 0:
		all_losses.append(current_loss / plot_every)
		current_loss = 0


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.show()


######## Evaluating ########

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


######## Running on User Input ########
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


predict('Dovesky')
predict('Jackson')
predict('Satoshi')