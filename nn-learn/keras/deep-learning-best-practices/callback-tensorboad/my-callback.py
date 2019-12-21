# callback还可以自己写, 只要是继承于 keras.callbacks.Callback 的子类
'''
函数(方法)
on_epoch_begin	->	Called at the start of every epoch
on_epoch_end	->	Called at the end of every epoch
on_batch_begin	->	Called right before processing each batch
on_batch_end	->	Called right after processing each batch
on_train_begin	->	Called at the start of training
on_train_end	->	Called at the end of training
'''
import tensorflow as tf
import numpy as np


class ActivationLogger(tf.keras.callbacks.Callback):
	# Called by the parent model before training, 
	# to inform the callback of what model will be calling it
	def set_model(self, model):
		self.model = model
		layer_outputs = [layer.output for layer in model.layers]
		# Model instance that returns the activations of every layer
		self.activations_model = tf.keras.models.Model(model.input, layer_outputs)


	def on_epoch_end(self, epoch, logs=None):
		if self.validation_data is None:
			raise RuntimeError('Requires validation_data.')
			
		# Obtains the first input sample of the validation data
		validation_sample = self.validation_data[0][0:1]
		activations = self.activations_model.predict(validation_sample)
		f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
		# Saves arrays to disk
		np.savez(f, activations)
		f.close()