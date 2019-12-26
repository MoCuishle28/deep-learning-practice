from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.dfm import DeepFactorizationMachineModel
import torch


fm = FactorizationMachineModel(field_dims=[10, 10, 10, 10, 10], embed_dim=2)
x = torch.tensor([[i for _ in range(5)] for i in range(10)])
print(x.shape)
y = fm(x)	# 最后通过 sigmoid 后输出的 fm
print(y)


dfm = DeepFactorizationMachineModel(field_dims=[10, 10, 10, 10, 10], embed_dim=2, mlp_dims=[10, 20, 5], dropout=0.5)
x = torch.tensor([[i for _ in range(5)] for i in range(10)])
print(x.shape)
y = dfm(x)	# 最后通过 sigmoid 后输出的 dfm
print(y)

# print('parameters:')
# for i, p in enumerate(fm.parameters()):
# 	print(p.shape)
# print('---')
# for i, p in enumerate(dfm.parameters()):
# 	print(i, p.shape)


optimizer = torch.optim.Adam(dfm.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()

def try_train():
	print('--------try train--------')

	def evaluate(predict):
		predict_list = []
		for prob in predict:
			if prob > 0.5:
				predict_list.append(1)
			else:
				predict_list.append(0)

		predict = torch.tensor(predict_list)
		precise = torch.sum(y.long() == predict).item()*1.0 / len(predict_list)
		print('precise:{}'.format(precise))
		return precise

	# 不行, 输入必须为整数
	# from sklearn.datasets.samples_generator import make_blobs
	# X, y = make_blobs(n_samples=1000, n_features=5, centers=[[-1 for _ in range(5)], [0 for _ in range(5)]], cluster_std=[0.4, 0.2], random_state =9)
	# print(X.shape, y.shape)

	import random
	import matplotlib.pyplot as plt

	half_size = 500
	X0 = torch.tensor([ [random.randint(0, 4) for _ in range(5)] for _ in range(half_size) ])
	y0 = torch.tensor([ 0 for _ in range(half_size)], dtype=torch.float32)
	X1 = torch.tensor([ [random.randint(5, 9) for _ in range(5)] for _ in range(half_size) ])
	y1 = torch.tensor([ 1 for _ in range(half_size)], dtype=torch.float32)

	X = torch.cat((X0, X1))
	y = torch.cat((y0, y1))
	print(X.shape, X.dtype, y.shape, y.dtype)
	
	loss_list = []
	precise_list = []

	for i in range(100):
		predict = dfm(X)
		# predict = fm(X)	# 参数没有变化
		
		loss = criterion(predict, y)
		print('LOSS:{:.4f}'.format(loss.item()))
		loss_list.append(loss.item())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i%10 == 0:
			precise_list.append(evaluate(predict))

	predict = dfm(X)
	predict_list = []
	for prob in predict:
		if prob > 0.5:
			predict_list.append(1)
		else:
			predict_list.append(0)

	predict = torch.tensor(predict_list)
	print(torch.sum(y.long() == predict).item()*1.0 / len(predict_list))

	plt.subplot(1, 3, 1)
	plt.title('LOSS')
	plt.xlabel('episode')
	plt.ylabel('LOSS')
	plt.plot(range(len(loss_list)), loss_list)

	plt.subplot(1, 3, 3)
	plt.title('Total Reward')
	plt.xlabel('episode')
	plt.ylabel('Total Reward')
	plt.plot(range(len(precise_list)), precise_list)
	plt.show()


try_train()