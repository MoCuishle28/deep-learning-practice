import tensorflow as tf


callbacks_list = [
	# 在验证集表现不再提升时提前停止 monitor='acc' 显示验证集的精度  patience=1 在1个 epoch 不再提升精度时就停止
	tf.keras.callbacks.EarlyStopping(monitor='acc', patience=1,), 
	# 每个 epoch 都保存一次模型, monitor='val_loss', save_best_only=True 若验证集 loss 不再下降, 则不覆盖文件
	tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', save_best_only=True,)
]

# 还没有构建 model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks_list, validation_data=(x_val, y_val))