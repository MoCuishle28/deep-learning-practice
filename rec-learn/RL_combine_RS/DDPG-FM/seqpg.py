import torch
import numpy as np


def gaussian_likelihood(x, mu, log_std):
	# 运算变量必须都是 tensor (np.pi)
	pre_sum = -0.5 * ( ((x-mu) / torch.exp(log_std))**2 + 2 * log_std + torch.log(torch.tensor([2*np.pi])) )
	return torch.sum(pre_sum)

# TODO