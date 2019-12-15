import os

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


data_dir = 'data/jena_climate_2009_2016'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
# ['"Date Time"', '"p (mbar)"', '"T (degC)"', '"Tpot (K)"', '"Tdew (degC)"', '"rh (%)"', '"VPmax (mbar)"', '"VPact (mbar)"', '"VPdef (mbar)"', '"sh (g/kg)"', '"H2OC (mmol/mol)"', '"rho (g/m**3)"', '"wv (m/s)"', '"max. wv (m/s)"', '"wd (deg)"']
header = lines[0].split(',')
# each line is a timestep: a record of a date and 14 weather-related values
lines = lines[1:]


float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
	values = [float(x) for x in line.split(',')[1:]]
	float_data[i, :] = values

# Normalizing the data (前200000条数据作为训练集)
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

# Generator yielding timeseries samples and their targets
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
	if max_index is None:
		max_index = len(data) - delay - 1
	i = min_index + lookback
	while 1:
		if shuffle:
			rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
		else:
			if i + batch_size >= max_index:
				i = min_index + lookback
			rows = np.arange(i, min(i + batch_size, max_index))
			i += len(rows)

		# (batch, )
		samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
		targets = np.zeros((len(rows),))
		for j, row in enumerate(rows):
			indices = range(rows[j] - lookback, rows[j], step)
			samples[j] = data[indices]
			targets[j] = data[rows[j] + delay][1]
		yield samples, targets


# Preparing the training, validation, and test generators
lookback = 1440		# Observations will go back 10 days.
step = 6			# Observations will be sampled at one data point per hour.  ???
delay = 144			# Targets will be 24 hours in the future.
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, 
	min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)

val_gen = generator(float_data, lookback=lookback, delay=delay, 
	min_index=200001, max_index=300000, step=step, batch_size=batch_size)

test_gen = generator(float_data, lookback=lookback, delay=delay, 
	min_index=300001, max_index=None, step=step, batch_size=batch_size)

# How many steps to draw from val_gen in order to see the entire validation set
val_steps = (300000 - 200001 - lookback)
# How many steps to draw from test_gen in order to see the entire test set
test_steps = (len(float_data) - 300001 - lookback)


######################## baseline ########################
# Computing the common-sense baseline MAE(mean absolute error)
def evaluate_naive_method():
	batch_maes = []
	for step in range(val_steps):
		samples, targets = next(val_gen)
		preds = samples[:, -1, 1]		# 总是预测 24 小时后的温度和当前一样
		mae = np.mean(np.abs(preds - targets))
		batch_maes.append(mae)
	print(np.mean(batch_maes)) 		# 0.2896994

# evaluate_naive_method()
# 因为已经归一化了,所以: 
celsius_mae = 0.29 * std[1]
print('common-sense baseline 误差:', celsius_mae, '℃')


# Training and evaluating a GRU-based model
def GRU_based():
	model = Sequential()

	# 先通过 1D convnet 再通过 RNN 这样更快(1D convnet 相当于先提取了特征, 这样后面 RNN 处理的特征更少了)
	# 但是损失为 nan ?????
	# model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
	# model.add(layers.MaxPooling1D(3))
	# model.add(layers.Conv1D(32, 5, activation='relu'))
	# model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))

	# 不通过1D convent 直接通过 RNN(损失也是 nan?????)
	model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, 
		return_sequences=True, input_shape=(None, float_data.shape[-1])))
	model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))

	model.add(layers.Dense(1))
	model.compile(optimizer=RMSprop(), loss='mae')
	# 十几分钟以上一个 epoch
	history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=5, 	# epochs=40
		validation_data=val_gen, validation_steps=val_steps)

	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(loss) + 1)
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

GRU_based()