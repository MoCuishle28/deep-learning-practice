import os

from tensorflow.keras.layers import Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# The Embedding layer takes at least two arguments:
# the number of possible tokens (here, 1,000: 1 + maximum word index)
# and the dimensionality of the embeddings (here, 64).
# embedding_layer = Embedding(1000, 64)

max_features = 10000	# 只考虑文本中最常见的 10000 个词
maxlen = 20				# 只考虑文本中最常见的词的前 20 个(后面直接截断)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)	# x_train.shape (25000, ); 内容是每个 word 的 index
# Turns the lists of integers into a 2D integer tensor of shape (samples, maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)	
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
# 输出 tensor 的 shape (samples, sequence_length, embedding_dimensionality) -> (samples, maxlen, 8)
# 输出的是每句话的每个词的 embedding (samples, maxlen, 8) -> (句子数, 每个句子词数, embedding)
model.add(Embedding(10000, 8, input_length=maxlen))
# Flattens the 3D tensor of embeddings into a 2D tensor of shape (samples, maxlen * 8) 将一句话展开成一个 dense vector
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# print(model.summary())

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# test embeddings
# embedding_layer = Embedding(10000, 8, input_length=maxlen)
# word_embedding = embedding_layer(x_train)
# print(word_embedding.shape)
# print(word_embedding)