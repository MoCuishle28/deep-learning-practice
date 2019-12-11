import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


imdb_dir = 'data\\aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
# print(train_dir)		# data\aclImdb\train

import datetime
start = datetime.datetime.now()

labels = []
texts = []
for label_type in ['neg', 'pos']:
	dir_name = os.path.join(train_dir, label_type)
	for fname in os.listdir(dir_name):
		if fname[-4:] == '.txt':
			with open(os.path.join(dir_name, fname), encoding='UTF-8') as f:
				texts.append(f.read())	
			# f = open(os.path.join(dir_name, fname), encoding='UTF-8')
			# texts.append(f.read())
			# f.close()

			if label_type == 'neg':
				labels.append(0)
			else:
				labels.append(1)


print('耗时:', (datetime.datetime.now()-start))
print(texts[0])