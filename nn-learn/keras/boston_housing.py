from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


# 各个数据量纲不同（有的 0~1, 有的 0~100, 需要归一化）
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape, test_data.shape)

mean = train_data.mean(axis=0)
train_data -= mean

std = train_data.std(axis=0)
train_data /= std

# Note 用 train data 的 mean 和 std 去给 test data 做归一化
# You should never use in your workflow any quantity computed on the test data, even for something as simple as data normalization
test_data -= mean
test_data /= std


def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))
	# loss func-> mean squared error, mean absolute error-> mae
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	return model


# 数据太少，直接分交叉验证集会导致交叉验证集很少数据，所以用 K-Fold 分交叉验证集
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_scores = []
all_mae_histories = []

for i in range(k):
	print('processing fold #', i)
	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

	# i*num_val_samples 之前的数据 cat (i+1)*num_val_samples 之后的数据（相当于中间空出 num_val_samples）
	partial_train_data = np.concatenate(
		[train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
	partial_train_targets = np.concatenate(
		[train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

	model = build_model()
	model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
	val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
	all_scores.append(val_mae)

	# 没有 val_mean_absolute_error
	# history = model.fit(partial_train_data, partial_train_targets, 
	# 	validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=0)
	# mae_history = history.history['val_mean_absolute_error']
	# all_mae_histories.append(mae_history)

print(all_scores)
print(np.mean(all_scores))

# print(len(all_mae_histories), len(all_mae_histories[0]))
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

'''
Replace each point with an exponential moving average of the previous points, 
to obtain a smooth curve.
'''
# def smooth_curve(points, factor=0.9):
# 	smoothed_points = []
# 	for point in points:
# 		if smoothed_points:
# 			previous = smoothed_points[-1]
# 			smoothed_points.append(previous * factor + point * (1 - factor))
# 		else:
# 			smoothed_points.append(point)
# 	return smoothed_points

# 忽略前 10 个
# smooth_mae_history = smooth_curve(average_mae_history[10:])

# plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()