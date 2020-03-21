import pickle
import argparse

import pandas as pd
import numpy as np


def save_obj(obj, name):
	with open('../../data/ml_1M_row/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open('../../data/ml_1M_row/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


def build_all_data_list():
	# uid:[[mrow, rating, timestamp], ...] 	value 内部要按照时间排序
	users_rating = load_obj('users_rating')
	all_data = []	# element -> (uid, mrow, rating, timestamp)
	for uid, behavior_list in users_rating.items():
		for behavior in behavior_list:
			element = [uid, behavior[0], behavior[1], behavior[2]]
			all_data.append(','.join([str(x) for x in element]))
	return all_data


def build_data_target(all_data):
	data = []
	target = []
	for idx, element in enumerate(all_data):
		data.append(element)
		if idx + 1 < len(all_data) and element[0] == all_data[idx + 1][0]:
			next_item = all_data[idx + 1]
			target.append(next_item[1])
		else:
			data.pop()

	return np.array(data), np.array(target)


def divide_dataset(data, target, ratio=1):
	dataset_list = [[], [], []]	# training, valid, testing
	target_list = [[], [], []]

	data, target = data.tolist(), target.tolist()
	tmp_data, tmp_target = [data[0]], [target[0]]
	for x, y in zip(data[1:], target[1:]):
		if tmp_data[-1][0] == x[0]:	# 同一个 user
			tmp_data.append(x)
			tmp_target.append(y)
		else:
			size = len(tmp_data)
			for idx in [-1, -2]:
				t1, t2 = [], []
				for i in range((size//10) * ratio):
					t1.append(tmp_data.pop())
					t2.append(tmp_target.pop())
				dataset_list[idx].extend(t1[::-1])
				target_list[idx].extend(t2[::-1])

			dataset_list[0].extend(tmp_data)
			target_list[0].extend(tmp_target)
			tmp_data, tmp_target = [], []
			tmp_data.append(x)
			tmp_target.append(y)

	train_data, train_target = np.array(dataset_list[0]), np.array(target_list[0])
	valid_data, valid_target = np.array(dataset_list[1]), np.array(target_list[1])
	test_data, test_target = np.array(dataset_list[2]), np.array(target_list[2])
	return train_data, train_target, valid_data, valid_target, test_data, test_target


def save_data(train_data, train_target, valid_data, valid_target, test_data, test_target):
	base_dir = '../../data/ml_1M_row/seq_predict/'
	np.save(base_dir + 'train_data.npy', train_data)
	np.save(base_dir + 'train_target.npy', train_target)
	np.save(base_dir + 'valid_data.npy', valid_data)
	np.save(base_dir + 'valid_target.npy', valid_target)
	np.save(base_dir + 'test_data.npy', test_data)
	np.save(base_dir + 'test_target.npy', test_target)


def main(args):
	if args.build_dataset == 'y':
		all_data_file = 'ml-1m.rating'

		if args.write_data_list == 'y':
			all_data = build_all_data_list()
			with open(all_data_file, 'w', encoding='utf8') as f:
				for element in all_data:
					f.writelines(element+'\n')

		all_data = []
		for line in open(all_data_file):
			element = line.strip().split(',')
			uid, mid, timestamp = int(element[0]), int(element[1]), int(element[3])
			rating = float(element[2])
			all_data.append([uid, mid])
		print('check!')
		print(all_data[:10])	# [uid, mid]
		print(all_data[-1])

		data, target = build_data_target(all_data)
		print('check!')
		print(data.shape, target.shape, data.dtype, target.dtype)
		print(data[:5])
		print(target[:5])

		train_data, train_target, valid_data, valid_target, test_data, test_target = divide_dataset(data, target)	# 默认 8:1:1
		save_data(train_data, train_target, valid_data, valid_target, test_data, test_target)

		print('shape:')
		print(train_data.shape, train_target.shape)
		print(valid_data.shape, valid_target.shape)
		print(test_data.shape, test_target.shape)
		print('check!')
		print(train_data)
		print(train_target)
		print('---')
		print(valid_data)
		print(valid_target)
		print('---')
		print(test_data)
		print(test_target)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Hyperparameters")
	parser.add_argument('--write_data_list', default="n")
	parser.add_argument('--build_dataset', default="y")

	args = parser.parse_args()
	main(args)