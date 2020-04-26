import os

import torch
import torch.nn as nn


class Net(nn.Module):
	def __init__(self, args, device):
		super(Net, self).__init__()
		self.args = args
		self.device = device
		self.fc0 = nn.Linear(args.seq_hidden_size, args.max_iid + 1)

	def forward(self, x):
		return self.fc0(x)


class REM(nn.Module):
	def __init__(self, args, device):
		super(REM, self).__init__()
		self.args = args
		self.device = device
		self.hidden_size = args.seq_hidden_size
		self.seq_layer_num = args.seq_layer_num

		self.i_embedding = nn.Embedding(args.max_iid + 1 + 1, args.i_emb_dim)	# 初始状态 mid=70852
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.gru = nn.GRU(args.i_emb_dim, self.hidden_size, self.seq_layer_num, batch_first=True, dropout=args.dropout if self.seq_layer_num > 1 else 0.0)
		self.ln = None
		if args.layer_trick == 'bn':
			self.ln = nn.BatchNorm1d(self.hidden_size, affine=True)
		elif args.layer_trick == 'ln':
			self.ln = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

		self.models = nn.ModuleList([Net(args, device) for _ in range(args.K)])
		self.alpha = nn.Parameter(torch.softmax(torch.ones(args.K, 1, dtype=torch.float32, device=device), dim=0))
		# nn.init.uniform_(self.alpha, a=0, b=1)


	def forward(self, x, is_train=True):
		x = self.i_embedding(x.long().to(self.device))
		x = x.view(x.shape[0], x.shape[1], -1)
		# 需要 requires_grad=True 吗？
		h0 = torch.zeros(self.seq_layer_num, x.size(0), self.hidden_size, device=self.device)
		out, _ = self.gru(x, h0)  	# out: tensor of shape (batch_size, seq_length, hidden_size)
		out = out[:, -1, :]			# 最后时刻的 seq 作为输出
		if self.args.layer_trick != 'none':
			out = self.ln(out)

		if is_train:
			rets = 0
			for i, model in enumerate(self.models):
				k = self.alpha[i]
				ret = model(out)
				rets += (ret * k)
			ret = rets
		else:
			# (K, batch, item_num)
			ret = torch.stack([model(out) for model in self.models]).to(self.device)
			ret = ret.mean(dim=0)
		return ret