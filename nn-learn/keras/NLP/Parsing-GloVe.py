import os
import datetime

import numpy as np


glove_dir = 'data/glove.6B'
embeddings_index = {}
start = datetime.datetime.now()

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='UTF-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('耗时:', (datetime.datetime.now()-start))
print('Found %s word vectors.' % len(embeddings_index))
print(embeddings_index['is'])