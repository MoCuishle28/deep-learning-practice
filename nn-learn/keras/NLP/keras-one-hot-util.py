import os
from tensorflow.keras.preprocessing.text import Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# Creates a tokenizer, configured to only take into account the 1,000 most common words\
# 如果 num_words 太小 而数据集包含的 word 太多，会导致很多次的 hash 冲突，多个 word 会用同一个 one-hot 表示
tokenizer = Tokenizer(num_words=1000)
# Builds the word index
tokenizer.fit_on_texts(samples)

# Turns strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)
# [[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]
print('sequences:', sequences)

# You could also directly get the one-hot binary representations.
# Vectorization modes other than one-hot encoding are supported by this tokenizer.
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
# [[0. 1. 1. ... 0. 0. 0.] [0. 1. 0. ... 0. 0. 0.]]
print('one_hot_results:', one_hot_results)

# How you can recover the word index that was computed
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
print(word_index)