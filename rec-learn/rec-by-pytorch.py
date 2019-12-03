import logging
import datetime

import pandas as pd
import numpy as np
import torch


logging.basicConfig(level = logging.DEBUG, filename = 'log/CF-log.log', filemode = 'a')
log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logging.debug('start time: '+ str(log_time))


userNo = 611
movieNo = 9742

rating = np.load('data/rating.npy')
record = rating > 0						# 大于0的地方表示已经有评分
record = np.array(record, dtype=int)	# 将布尔值转为 0/1  0表示未评分,1表示已评分


# 假设有10种类型的电影
num_features = 10
LR = 1e-3
EPOCH = 1000

record = torch.tensor(record, dtype=torch.float64)
rating = torch.tensor(rating, dtype=torch.float64)


# 初始化电影内容矩阵 X; 用户矩阵 θ
# X_parameter = torch.tensor(np.random.normal(0.0, 0.35, (movieNo, num_features)))	# 随机初始化正太分布的 X
# Theta_parameter = torch.tensor(np.random.normal(0.0, 0.35, (userNo, num_features)))
# X_parameter.requires_grad_(requires_grad=True)
# Theta_parameter.requires_grad_(requires_grad=True)


# 直接加载
X_parameter = torch.tensor(np.load('models/X_parameter_withoutNorm.npy'))
X_parameter.requires_grad_(requires_grad=True)

Theta_parameter = torch.tensor(np.load('models/Theta_parameter_withoutNorm.npy'))
Theta_parameter.requires_grad_(requires_grad=True)


# 定义代价函数 乘以record是用0代替未评分电影
def Loss(predict_rating):
	return 1/2 * torch.sum(((predict_rating - rating) * record)**2) + 1/2 * (torch.sum(X_parameter**2) + torch.sum(Theta_parameter**2))

optimizer = torch.optim.RMSprop([X_parameter, Theta_parameter], lr=LR)

for i in range(EPOCH):
	predict_rating = torch.mm(X_parameter, Theta_parameter.t())
	loss = Loss(predict_rating)

	if i % 100 == 0:
		print('EPOCH:', i,'Loss:', loss)
		logging.debug('Loss: '+ str(loss))

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

# 保持参数
np.save('models/X_parameter_withoutNorm.npy', X_parameter.detach().numpy())
np.save('models/Theta_parameter_withoutNorm.npy', Theta_parameter.detach().numpy())


predicts = torch.mm(X_parameter, Theta_parameter.t()).detach().numpy()

errors = np.sqrt(np.sum((predicts - rating.detach().numpy()**2)))
print('Error:', errors)

logging.debug('ERROR: '+ str(errors))

log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logging.debug('end time: '+ str(log_time))