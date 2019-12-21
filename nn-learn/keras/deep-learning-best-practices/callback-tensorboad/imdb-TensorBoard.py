import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


max_features = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = tf.keras.models.Sequential()

model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Before you start using TensorBoard, you need to create a directory where you’ll store the log files it generates.
# mkdir my_log_dir
callbacks = [tf.keras.callbacks.TensorBoard(
	# Log files will be written at this location
	log_dir='my_log_dir', 
	# Records activation histograms every 1 epoch
	histogram_freq=1, 
	# Records embedding data every 1 epoch
	embeddings_freq=1,)]
	# The Embeddings tab gives you a way to inspect the embedding locations 
	# and spatial relationships of the 10,000 words in the input vocabulary
	# tensorboard 会自动将 embedding 降维到2D、3D, 用户可以查看 embedding 的关系

history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=callbacks)

# 运行以下指令查看 tensorboard
# tensorboard --logdir=my_log_dir
# http://localhost:6006

# 画出网络结构
from tensorflow.keras.utils import plot_model
# Using it requires that you’ve installed the Python pydot and pydot-ng libraries as well as the graphviz library. 
# show_shapes=True -> displaying shape information
plot_model(model, show_shapes=True, to_file='model.png')