import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2		# Dimensionality of the latent space: a 2D plane

input_img = tf.keras.Input(shape=img_shape)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

# The input image ends up being encoded into these two parameters
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# using z_mean and z_log_var , the parameters of the statistical distribution 
# assumed to have produced input_img , to generate a latent space point z 
def sampling(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
	return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# decoder implementation
# Input where you’ll feed z
decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
# Reshapes z into a feature map of the same shape as the feature map just before the last Flatten layer in the encoder model
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
decoder = Model(decoder_input, x)
z_decoded = decoder(z)


# Custom layer used to compute the VAE loss
class CustomVariationalLayer(tf.keras.layers.Layer):
	def vae_loss(self, x, z_decoded):
		x = K.flatten(x)
		z_decoded = K.flatten(z_decoded)
		xent_loss = tf.keras.metrics.binary_crossentropy(x, z_decoded)
		kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return K.mean(xent_loss + kl_loss)

	# You implement custom layers by writing a call method
	def call(self, inputs):
		x = inputs[0]
		z_decoded = inputs[1]
		loss = self.vae_loss(x, z_decoded)
		self.add_loss(loss, inputs=inputs)
		return x 		# layer must return something


# Calls the custom layer on the input and the decoded output to obtain the final model output
y = CustomVariationalLayer()([input_img, z_decoded])


# Training the VAE
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)

(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))

# the loss is taken care of in the custom layer, you don’t specify an external loss at compile time
vae.fit(x=x_train, y=None, shuffle=True, epochs=10, batch_size=batch_size, validation_data=(x_test, None))


import matplotlib.pyplot as plt
from scipy.stats import norm

n = 15		# 生成 15X15 张照片
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
	for j, xi in enumerate(grid_y):
		z_sample = np.array([[xi, yi]])
		z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
		x_decoded = decoder.predict(z_sample, batch_size=batch_size)
		digit = x_decoded[0].reshape(digit_size, digit_size)
		figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()