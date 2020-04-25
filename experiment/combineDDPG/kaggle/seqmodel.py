import torch
import torch.nn as nn


class SeqModel(nn.Module):
	def __init__(self, args, device):
		super(SeqModel, self).__init__()
		self.args = args
		self.device = device
		self.hidden_size = args.seq_hidden_size
		self.seq_layer_num = args.seq_layer_num

		self.m_embedding = nn.Embedding(args.max_iid + 1 + 1, args.m_emb_dim)	# 初始状态 mid=70852
		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.gru = nn.GRU(args.m_emb_dim, self.hidden_size, self.seq_layer_num, batch_first=True, dropout=args.dropout if self.seq_layer_num > 1 else 0.0)
		self.ln = None
		if args.layer_trick == 'bn':
			self.ln = nn.BatchNorm1d(self.hidden_size, affine=True)
		elif args.layer_trick == 'ln':
			self.ln = nn.LayerNorm(self.hidden_size, elementwise_affine=True)		
		self.fc = nn.Linear(self.hidden_size, args.seq_output_size)

	
	def forward(self, x):
		# input x:(1, 10, 1)/(batch, 10), output x->(1, 10, 1, 128)/(batch, 10, 128)
		x = self.m_embedding(x.long().to(self.device))
		x = x.view(x.shape[0], x.shape[1], -1)
		# 需要 requires_grad=True 吗？
		h0 = torch.zeros(self.seq_layer_num, x.size(0), self.hidden_size, device=self.device)

		out, _ = self.gru(x, h0)  	# out: tensor of shape (batch_size, seq_length, hidden_size)
		out = out[:, -1, :]			# 最后时刻的 seq 作为输出
		if self.args.layer_trick != 'none':
			out = self.ln(out)
		out = self.fc(out)
		return out