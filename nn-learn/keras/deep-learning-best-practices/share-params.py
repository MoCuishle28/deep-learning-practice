from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model


# 实例化一个 LSTM layer
lstm = layers.LSTM(32)

# 将 lstm 用在两个输入上 (share)
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)

model = Model([left_input, right_input], predictions)
# 还没有训练数据 left_data, right_data, targets
model.fit([left_data, right_data], targets)