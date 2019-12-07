from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = load_model('models/cats_and_dogs_small_2.h5')
# print(model.summary())

img_path = 'data/test/cats/cat.1700.jpg'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# Its shape is (1, 150, 150, 3)
# print(img_tensor.shape)

# display pic
# plt.imshow(img_tensor[0])
# plt.show()

# 前 8 个层的输出
layer_outputs = [layer.output for layer in model.layers[:8]]
# 一个输入，8个输出的 activation_model
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)
# print(len(activations))		# 8

first_layer_activation = activations[0]
# print(first_layer_activation.shape)	# (1, 148, 148, 32) 第一层的输出

# 只画第4个通道
# This channel appears to encode a diagonal edge detector.
# plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
# plt.show()

# 5 channel		不同的 channel 对应不同的特征提取
# plt.matshow(first_layer_activation[0, :, :, 5], cmap='viridis')
# plt.show()


# Names of the layers, so you can have them as part of your plot
layer_names = []
for layer in model.layers[:8]:
	layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
	n_features = layer_activation.shape[-1]		# channel
	size = layer_activation.shape[1]			# 长和宽

	# 一行摆几个图片？
	n_cols = n_features // images_per_row
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	
	for col in range(n_cols):
		for row in range(images_per_row):
			channel_image = layer_activation[0,:, :, col * images_per_row + row]

			channel_image -= channel_image.mean()
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

	scale = 1. / size
	plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
	plt.title(layer_name)
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.show()