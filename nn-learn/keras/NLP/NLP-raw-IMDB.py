import os
import argparse

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
# dest='train' 可以通过 args.train 来访问
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')
args = parser.parse_args()


imdb_dir = 'data\\aclImdb'			# 情感分析数据
train_dir = os.path.join(imdb_dir, 'train')
# print(train_dir)		# data\aclImdb\train

import datetime
start = datetime.datetime.now()

labels = []
texts = []
for label_type in ['neg', 'pos']:
	dir_name = os.path.join(train_dir, label_type)
	for fname in os.listdir(dir_name):
		if fname[-4:] == '.txt':
			with open(os.path.join(dir_name, fname), encoding='UTF-8') as f:
				texts.append(f.read())	

			if label_type == 'neg':
				labels.append(0)
			else:
				labels.append(1)


print('读取数据耗时:', (datetime.datetime.now()-start))
print(len(texts))	# 25000
# print(texts[0])		# 一段文本


maxlen = 100					# Cuts off reviews after 100 words
training_samples = 200			# Trains on 200 samples
validation_samples = 10000		# Validates on 10,000 samples
max_words = 10000				# Considers only the top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)						# Builds the word index
sequences = tokenizer.texts_to_sequences(texts)		# Turns strings into lists of integer indices

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Splits the data into a training set and a validation set, but first shuffles the data,
# because you’re starting with data in which samples are ordered (all negative first, then all positive)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
# 打乱 data, 原来是 neg 全在一堆, pos 全在一堆的
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]	# 前200个
y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


glove_dir = 'data/glove.6B'
embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='UTF-8')
for line in f:
	values = line.split()
	word = values[0]		# key 是词 (string)
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


######### Preparing the GloVe word-embeddings matrix #########
embedding_dim = 100
# 可以构建一个 matrix shape (max_words, embedding_dim)
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
	if i < max_words:
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# Words not found in the embedding index will be all zeros.
			embedding_matrix[i] = embedding_vector


model = Sequential()
# 加入 embedding 层, 再装载 pre-train 的 embedding 矩阵
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# print(model.summary())

# Loading pretrained word embeddings into the Embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False	# freeze the Embedding layer

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

if args.train:
	history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

	# model.save_weights('../models/pre_trained_glove_model.h5')

	# plot
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)

	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()


################# Tokenizing the data of the test set #################
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []
for label_type in ['neg', 'pos']:
	dir_name = os.path.join(test_dir, label_type)
	for fname in sorted(os.listdir(dir_name)):
		if fname[-4:] == '.txt':
			f = open(os.path.join(dir_name, fname), encoding='UTF-8')
			texts.append(f.read())
			f.close()
			if label_type == 'neg':
				labels.append(0)
			else:
				labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

# Next, load and evaluate the first model.
model.load_weights('../models/pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)