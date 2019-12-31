import pickle

import numpy as np


def save_obj(obj, name):
	with open('../data/ml20/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open('../data/ml20/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


def valid_mid_map_one_hot(mid_map_one_hot, data):
	yep = True
	for vecotr in data:
		mask = (mid_map_one_hot[vecotr[1]] == vecotr[1:])
		if np.sum(mask) != mask.shape[0]:
			print(vecotr[1])
			yep = False
			break
	return yep


def main():
	data = np.load('../data/ml20/mini_data.npy')
	print(data.shape)

	mid_map_one_hot = {}	# mid: one-hot 20 维度 (除了 uid)


	for vecotr in data:
		mid_map_one_hot[vecotr[1]] = vecotr[1:]

	yep = valid_mid_map_one_hot(mid_map_one_hot, data)

	if yep:
		print('yep!')
		save_obj(mid_map_one_hot, 'mini_ml20_mid_map_one_hot')
	else:
		print('no!!!')

	mid_map_one_hot = load_obj('mini_ml20_mid_map_one_hot')

	yep = valid_mid_map_one_hot(mid_map_one_hot, data)
	print('yep!' if yep else 'no!!!')



if __name__ == '__main__':
	main()