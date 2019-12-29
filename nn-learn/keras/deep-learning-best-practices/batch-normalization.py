
# The BatchNormalization layer takes an axis argument, 
# which specifies the feature axis that should be normalized.
# This argument defaults to -1
# This is the correct value when using Dense layers, Conv1D layers, RNN layers,
# and Conv2D layers with data_format set to "channels_last" . 
# 书 P261
conv_model.add(layers.Conv2D(32, 3, activation='relu'))
conv_model.add(layers.BatchNormalization())
dense_model.add(layers.Dense(32, activation='relu'))
dense_model.add(layers.BatchNormalization())