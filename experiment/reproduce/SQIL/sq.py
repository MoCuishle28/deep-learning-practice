import os

import torch
import torch.nn as nn


def parse_layers(layers, activative_func, layer_trick, p, output_size):
	params = []
	layers = [int(x) for x in layers.split(',')]
	for i, num in enumerate(layers[:-1]):
		params.append(nn.Linear(num, layers[i + 1]))
		if layer_trick != None:
			params.append(layer_trick(layers[i + 1]))
		params.append(activative_func)
		params.append(nn.Dropout(p=p))
	params.append(nn.Linear(layers[-1], output_size))
	return params


def get_activative_func(act):
	activative_func_dict = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leaky':nn.LeakyReLU(), 
		'selu':nn.SELU(), 'prelu':nn.PReLU(), 'tanh':nn.Tanh()}
	return activative_func_dict.get(act, nn.ReLU())


class SoftQ(nn.Module):
	def __init__(self, args, device):
		super(SoftQ, self).__init__()
		self.args = args
		self.device = device
		self.seq_input_size = args.m_emb_dim + args.g_emb_dim
		self.hidden_size = args.seq_hidden_size
		self.seq_layer_num = args.seq_layer_num

		self.u_embedding = nn.Embedding(args.max_uid + 1, args.u_emb_dim)
		self.m_embedding = nn.Embedding(args.max_mid + 1 + 1, args.m_emb_dim)	# 初始状态 mid=9742
		self.g_embedding = nn.Linear(args.feature_size - 2, args.g_emb_dim)

		dropout = args.dropout if self.seq_layer_num > 1 else 0.0
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.gru = nn.GRU(self.seq_input_size, self.hidden_size, self.seq_layer_num, batch_first=True, dropout=dropout)
		self.ln = None
		if args.layer_trick == 'bn':
			self.ln = nn.BatchNorm1d(self.hidden_size, affine=True)
		elif args.layer_trick == 'ln':
			self.ln = nn.LayerNorm(self.hidden_size, elementwise_affine=True)		
		self.fc = nn.Linear(self.hidden_size + args.u_emb_dim, args.max_mid + 1)

	
	def forward(self, x):
		'''
		x: (batch, hw, feature_size)
		return: (batch, args.max_mid + 1)
		'''
		uids = x[:, 0, -self.args.feature_size]
		mids = x[:, :, -(self.args.feature_size - 1)]
		genres = x[:, :, -(self.args.feature_size - 2):]

		uemb = self.u_embedding(uids.long().to(self.device))
		memb = self.m_embedding(mids.long().to(self.device))
		gemb = self.g_embedding(genres.to(self.device))
		x = torch.cat([memb, gemb], -1).to(self.device)
		h0 = torch.zeros(self.seq_layer_num, x.size(0), self.hidden_size, device=self.device)

		out, _ = self.gru(x, h0)  	# out: tensor of shape (batch_size, seq_length, hidden_size)
		out = out[:, -1, :]			# 最后时刻的 seq 作为输出
		if self.args.layer_trick != 'none':
			out = self.ln(out)
		out = torch.cat([uemb, out], -1)
		out = self.fc(out)
		return out