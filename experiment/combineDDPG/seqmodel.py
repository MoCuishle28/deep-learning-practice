import torch
import torch.nn as nn


class SeqModel(nn.Module):
	def __init__(self, args, device):
		super(SeqModel, self).__init__()
		self.args = args
		self.device = device
		self.seq_input_size = args.m_emb_dim + args.g_emb_dim
		self.hidden_size = args.seq_hidden_size
		self.seq_layer_num = args.seq_layer_num

		# embedding layer
		self.u_embedding = nn.Embedding(args.max_uid + 1, args.u_emb_dim)
		self.m_embedding = nn.Embedding(args.max_mid + 1 + 1, args.m_emb_dim)	# 刚开始时 mid=9742
		self.g_embedding = nn.Linear(args.fm_feature_size - 3, args.g_emb_dim)

		# batch_first = True 则输入输出的数据格式为 (batch, seq, feature)
		self.gru = nn.GRU(self.seq_input_size, self.hidden_size, self.seq_layer_num, batch_first=True)
		if args.norm_layer == 'bn':
			self.ln1 = nn.BatchNorm1d(self.hidden_size, affine=True)
		elif args.norm_layer == 'ln':
			self.ln1 = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

		self.fc = nn.Linear(self.hidden_size + args.u_emb_dim, args.seq_output_size)


	def forward(self, x):
		'''
		x: (batch, seq_len, feature_size)
		return: (batch, args.seq_output_size)
		'''
		uids = x[:, 0, -self.args.fm_feature_size]
		mids = x[:, :, -(self.args.fm_feature_size - 1)]
		genres = x[:, :, -(self.args.fm_feature_size - 2):-1]
		clicked = x[:, :, -1]

		uemb = self.u_embedding(uids.long().to(self.device))
		memb = self.m_embedding(mids.long().to(self.device))
		gemb = self.g_embedding(genres.to(self.device))

		x = torch.cat([memb, gemb], -1).to(self.device)
		h0 = torch.zeros(self.seq_layer_num, x.size(0), self.hidden_size, device=self.device)
		
		out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
		out = out[:, -1, :]		# 最后时刻的 seq 作为输出
		if self.args.norm_layer != 'none':
			out = self.ln1(out)
		out = torch.cat([uemb, out], -1).to(self.device)
		out = self.fc(out)
		return out