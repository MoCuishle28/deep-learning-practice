import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 数据160多M... 破电脑跑不了


latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

# generator maps the input of shape (latent_dim,) into an image of shape (32, 32, 3)
generator = tf.keras.models.Model(generator_input, x)

# The GAN discriminator network
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)
discriminator = tf.keras.models.Model(discriminator_input, x)

discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')


# If the discriminator weights could be updated during this process, 
# then you’d be training the discriminator to always predict “real”
discriminator.trainable = False			# why???

gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.models.Model(gan_input, gan_output)

gan_optimizer = tf.keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

x_train = x_train[y_train.flatten() == 6]		# Selects frog images (class 6)
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

iterations = 1000
batch_size = 20
save_dir = '../data/'
start = 0

for step in range(iterations):
	random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

	generated_images = generator.predict(random_latent_vectors)

	stop = start + batch_size
	real_images = x_train[start: stop]
	combined_images = np.concatenate([generated_images, real_images])

	# Assembles labels, discriminating real from fake images (为什么 generated_images 是 1??? TODO)
	labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

	# Adds random noise to the labels—an important trick!
	labels += 0.05 * np.random.random(labels.shape)

	# Trains the discriminator 	(还是要训练的, 只是在训练 generator 的时候不能训练)
	d_loss = discriminator.train_on_batch(combined_images, labels)

	random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
	# Assembles labels that say “these are all real images” (it’s a lie!)
	misleading_targets = np.zeros((batch_size, 1))
	a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
	start += batch_size

	if start > len(x_train) - batch_size:
		start = 0
	if step % 100 == 0:
		gan.save_weights('gan.h5')
		print('discriminator loss:', d_loss)
		print('adversarial loss:', a_loss)
		img = image.array_to_img(generated_images[0] * 255., scale=False)
		img.save(os.path.join(save_dir,'generated_frog' + str(step) + '.png'))
		img = image.array_to_img(real_images[0] * 255., scale=False)
		img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))