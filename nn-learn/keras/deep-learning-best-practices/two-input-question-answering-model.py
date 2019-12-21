from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')
# Embeds the inputs into a sequence of vectors of size 64
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input) 
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# feeding data to multi-input model
num_samples = 1000
max_length = 100		# question 的最大长度

# 随机生成的数据
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))

# 报错： Function call stack: distributed_function

# 两种方法 fit
# Fitting using a list of inputs
# model.fit([text, question], answers, epochs=10, batch_size=128)

# Fitting using a dictionary of inputs (only if inputs are named)
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)