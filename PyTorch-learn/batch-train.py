import torch
import torch.utils.data as Data 	# 进行小批次训练的模块


BATCH_SIZE = 5	# 每批5个数据
# BATCH_SIZE = 8	# 若没批是8 则剩下那次不足的把用剩下的全用上


x = torch.linspace(1, 10, 10) 	# 1~10 的 10 个点
y = torch.linspace(10, 1, 10)	# 10~1 的 10 个点


# 定义一个数据集, 训练数据的是 第一个参数, 计算误差用第二个参数有 ?
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
	dataset = torch_dataset,
	batch_size = BATCH_SIZE,
	shuffle = True, 	# 训练时每个批次随机打乱数据
	# num_workers = 2,	# 每次提取用两个线程？(加快效率) win下不支持
)

print(x)
print(y)

# 整批数据训练3次
for epoch in range(3):
	for step, (batch_x, batch_y) in enumerate(loader):
		# 模拟 training...
		print('Epoch: ', epoch, '| Step: ', step, 
			'| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())