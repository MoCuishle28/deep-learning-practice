import pickle
import argparse

import torch


def load_obj(name):
	with open('../../data/new_ml_1M/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


class Emb(torch.nn.Module):
	def __init__(self, arg):
		super(Emb, self).__init__()
		self.arg = arg
		self.embedding = torch.nn.Embedding(arg.one_hot_size, arg.emb_size)


	def forward(self, x):
		return self.embedding(x)


def main():
	parser = argparse.ArgumentParser(description="Embedding Test")
	parser.add_argument('--one_hot_size', type=int, default=5)	# 可用的 idx: 0~4
	parser.add_argument('--emb_size', type=int, default=3)	# emb dim = 3
	args = parser.parse_args()

	emb = Emb(args)
	# print(emb.embedding.weight)
	# print(emb.embedding.weight.shape)
	a = [
		[0, 1, 2],
		[3, 4, 4]
	]
	b = [
		[2,2,2],
		[3,2,1]
	]

	data = torch.stack([torch.tensor(a), torch.tensor(b)])
	print(data.shape)	# [2, 2, 3] (batch, seq, feature size)
	ret = emb(data)
	print(ret)
	print(ret.shape)	# [2, 2, 3, 3]

	data = torch.tensor(a)
	ret = emb(data)	
	print(data.shape)	# [2, 3]
	print(ret)
	print(ret.shape)	# [2, 3, 3]

	a = [[1], [2], [3]]
	data = torch.tensor(a)
	ret = emb(data)
	print(ret)
	print(ret.shape, data.shape)


if __name__ == '__main__':
	main()