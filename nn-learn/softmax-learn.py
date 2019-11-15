import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

"""
torchvision包，它是服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。

torchvision主要由以下几部分构成：
	torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
	torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
	torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
	torchvision.utils: 其他的一些有用的方法。
"""

# 获取数据
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

# feature, label = mnist_train[0]
# print(feature.shape, label)  # Channel x Height x Width


# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))


# batch_size = 256
# if sys.platform.startswith('win'):
#     num_workers = 0  # 0表示不用额外的进程来加速读取数据
# else:
#     num_workers = 4
# # 创建一个读取小批量数据样本的DataLoader实例。
# train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 查看读取一次训练数据的时间
# start = time.time()
# for X, y in train_iter:
#     continue
# print('%.2f sec' % (time.time() - start))

num_inputs = 784
num_outputs = 10

# class LinearNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super().__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)


#     def forward(self, x): # x shape: (batch, 1, 28, 28)
#         y = self.linear(x.view(x.shape[0], -1))
#         return F.softmax(y, dim=1)


# net = LinearNet(num_inputs, num_outputs)

from collections import OrderedDict
# 更方便定义神经网络
net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', d2l.FlattenLayer()),	# 将 28x28 展开为 784 维
        ('linear', nn.Linear(num_inputs, num_outputs)),
        # 加了softmax反而准确率下降了 10% 为什么？
        # ('softmax', nn.Softmax(dim=1))
    ])
)


init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0) 

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)