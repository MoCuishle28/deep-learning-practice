from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
# from tensorflow.keras import metrics
import numpy as np
import matplotlib.pyplot as plt


# 数据集中只保留前 10000 个频率最高的词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 获取字典 (key-> word, value-> index (从1开始) )
# word_index = imdb.get_word_index()
# print(min(word_index.values()))

# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# indices are offset by 3 because 0, 1, 2 are reserved indices for “padding,” “start of sequence,” and “unknown.”
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decoded_review)


# 转为 one-hot
def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train.shape, x_test.shape)

# 向量化 labels (0 代表负面情感  1 代表正面情感)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


model = models.Sequential()
# 加上 regularizers
# 还可以同时 L1 L2  -> regularizers.l1_l2(l1=0.001, l2=0.001)
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))		# 加 Dropout
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001), 
	loss=losses.binary_crossentropy, 
	metrics=['acc'])


# validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# 训练 + 交叉验证
# history is a dictionary containing data about everything that happened during training
# history 可以方便画图
history = model.fit(partial_x_train, partial_y_train, 
	epochs=4, batch_size=512, validation_data=(x_val, y_val))
# print(history.history.keys())


history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

# 画图
# epochs = range(1, len(loss_values) + 1)
# # bo is blue dot
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# # b is blue line
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # clear the figure
# plt.clf()

# acc_values = history_dict['acc']
# val_acc_values = history_dict['val_acc']

# plt.plot(epochs, acc_values, 'r', label='Training acc')
# plt.plot(epochs, val_acc_values, 'b', label='Validation acc')

# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Acc')
# plt.legend()
# plt.show()


results = model.evaluate(x_test, y_test)
# 0->loss, 1->accuracy
print('results', results)

# 输出对新数据的预测
print(model.predict(x_test))