# You can use this callback to reduce the learning rate when the validation loss has stopped improving.
import tensorflow as tf


callbacks_list = [tf.keras.callbacks.ReduceLROnPlateau(
	# 根据验证集 loss 作为触发判断条件
	monitor='val_loss', 
	# 每次触发将 lr 除以 10
	factor=0.1, 
	# 10个 epoch 后 val_loss 仍然没有提升(下降)则触发
	patience=10,)]

# 尚未构建 model
model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks_list, validation_data=(x_val, y_val))