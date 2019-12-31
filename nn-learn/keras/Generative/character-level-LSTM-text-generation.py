import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import datetime
import random
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


start = datetime.datetime.now()
path = '../data/nietzsche.txt'

text = open(path).read().lower()
print('Corpus length:', len(text))


############## Vectorizing sequences of characters ##############
maxlen = 60			# extract sequences of 60 characters.
step = 3			# sample a new sequence every three characters.	(移动步伐)
sentences = []		# Holds the extracted sequences
next_chars = []		# Holds the targets	(后一个字符)

for i in range(0, len(text) - maxlen, step):
	sentences.append(text[i: i + maxlen])
	next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

chars = sorted(list(set(text)))
print('Unique characters:', len(chars))

char_indices = dict((char, chars.index(char)) for char in chars)
print('Vectorization...')

# (序列数(多少条序列), 序列长度, 字符数(多少个字符))
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		x[i, t, char_indices[char]] = 1
		y[i, char_indices[next_chars[i]]] = 1

print(x.shape, y.shape)		# (200278, 60, 58) 	(200278, 58)
print('耗时:{}'.format(datetime.datetime.now() - start))

model = tf.keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# Function to sample the next character given the model’s predictions
def sample(preds, temperature=1.0):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


for epoch in range(1, 3):
	print('epoch', epoch)
	model.fit(x, y, batch_size=128, epochs=1)		# Fits the model for one iteration on the data

	# Selects a text seed at random
	start_index = random.randint(0, len(text) - maxlen - 1)
	generated_text = text[start_index: start_index + maxlen]
	print('--- Generating with seed: "' + generated_text + '"')

	for temperature in [0.2, 0.5, 1.0, 1.2]:		# Tries a range of different sampling temperatures
		print('------ temperature:', temperature)
		sys.stdout.write(generated_text)
		# Generates 400 characters, starting from the seed text
		for i in range(400):
			sampled = np.zeros((1, maxlen, len(chars)))
			for t, char in enumerate(generated_text):
				sampled[0, t, char_indices[char]] = 1.

			preds = model.predict(sampled, verbose=0)[0]
			next_index = sample(preds, temperature)
			next_char = chars[next_index]
			generated_text += next_char
			generated_text = generated_text[1:]
			sys.stdout.write(next_char)
