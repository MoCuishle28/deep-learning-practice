import torch
from torch import nn
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import sys


# def use_svg_display():
	# 用矢量图显示 (没有 display模块)
	# display.set_matplotlib_formats('svg')

def set_figsize(figsize=(5.5, 3.5)):
	# use_svg_display()
	# 设置图的尺寸
	plt.rcParams['figure.figsize'] = figsize


# 本函数已保存在d2lzh包中方便以后使用
# 它每次返回batch_size（批量大小）个随机样本的特征和标签。
def data_iter(batch_size, features, labels):
	num_examples = len(features)
	indices = list(range(num_examples))		# 存 features 的 index
	random.shuffle(indices) 	 			# 样本的读取顺序是随机的
	for i in range(0, num_examples, batch_size):
		j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
		# 第一个参数0代表按行索引，1代表按列索引
		# 第二个参数是list 每个元素是返回的索引
		yield  features.index_select(0, j), labels.index_select(0, j)


def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
	return torch.mm(X, w) + b   # 矩阵乘法


# 定义损失函数
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
	# 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
	return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
	for param in params:
		param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data



"""
Fashion-MNIST中一共包括了10个类别:
	分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、
	sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）
"""
# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
				   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(i)] for i in labels]


# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
	# d2l.use_svg_display()
	# 这里的_表示我们忽略（不使用）的变量
	_, figs = plt.subplots(1, len(images), figsize=(8, 5))
	for f, img, lbl in zip(figs, images, labels):
		f.imshow(img.view((28, 28)).numpy())
		f.set_title(lbl)
		f.axes.get_xaxis().set_visible(False)
		f.axes.get_yaxis().set_visible(False)
	plt.show()


def load_data_fashion_mnist(batch_size):
	mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
	mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
	if sys.platform.startswith('win'):
		num_workers = 0  # 0表示不用额外的进程来加速读取数据
	else:
		num_workers = 4
	# 创建一个读取小批量数据样本的DataLoader实例。
	train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	return train_iter, test_iter


# 该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval() 				# 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train() 			# 改回训练模式
        else: # 自定义的模型
            if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
        n += y.shape[0]
    return acc_sum / n



# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
			  params=None, lr=None, optimizer=None):
	for epoch in range(num_epochs):
		train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
		for X, y in train_iter:
			y_hat = net(X)
			l = loss(y_hat, y).sum()

			# 梯度清零
			if optimizer is not None:
				optimizer.zero_grad()
			elif params is not None and params[0].grad is not None:
				for param in params:
					param.grad.data.zero_()

			l.backward()
			if optimizer is None:
				d2l.sgd(params, lr, batch_size)
			else:
				optimizer.step()  # “softmax回归的简洁实现”一节将用到

			# item() 是获取元素值（原来 l 是一个 1维的 tensor）
			train_l_sum += l.item()
			train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
			n += y.shape[0]

		# 每个 epoch 结束后计算一次在 test 集上的表现
		test_acc = evaluate_accuracy(test_iter, net)
		print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
			  % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


# 本函数已保存在d2lzh_pytorch包中方便以后使用
class FlattenLayer(nn.Module):
	def __init__(self):
		super(FlattenLayer, self).__init__()


	def forward(self, x): # x shape: (batch, *, *, ...)
		return x.view(x.shape[0], -1)


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
			 legend=None, figsize=(5.5, 3.5)):
	set_figsize(figsize)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.semilogy(x_vals, y_vals)
	if x2_vals and y2_vals:
		plt.semilogy(x2_vals, y2_vals, linestyle=':')
		plt.legend(legend)
	plt.show()