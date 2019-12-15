import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs


# X为样本特征，Y为样本簇类别， 共1000个样本，
# 每个样本4个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1],[2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, 
	centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state =9)
# plt.scatter(X[:, 0], X[:, 1], marker='o')
# plt.show()

print(X.shape)	# (1000, 2)


from sklearn.cluster import KMeans

y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
# print(y_pred)
# print(y)

# 不能这样评价!!! 因为虽然都分为3类, 但 y_pred 中, 哪个是0 哪个是1 哪个是2 不一定和 y 一样
# right = np.sum(y == y_pred) / 1000
# print('right rate:{}%'.format(right*100))

# 占用1行, 分两个子图, 占用第一个子图 -> 1, 2, 1
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

print(metrics.calinski_harabasz_score(X, y_pred))
#用Calinski-Harabasz Index评估四分类的聚类分数