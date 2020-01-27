import os
from zlib import crc32

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HOUSING_PATH = os.path.join("../datasets", "housing")


def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, "housing.csv")
	return pd.read_csv(csv_path)


# 直接划分训练集和测试集
def split_train_test(data, test_ratio):
	'''
	data: load data pd
	return: train data pd, test data pd
	'''
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data) * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]


# 用 hash 划分数据集、训练集, 保证多次运行或添加新数据时, 曾经进入训练集的数据不会进入测试集 (how?)
# 原来进入训练集的数据不会再变, 新加入的数据可能会挤掉测试集数据进入测试集, 因此保证了训练集数据不会进入测试集? (若使用 index 计算 hash 要保证新数据在末尾加入的情况下)
def test_set_check(identifier, test_ratio):
	# 根据 hash 找数据索引
	return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
	ids = data[id_column]
	in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
	return data.loc[~in_test_set], data.loc[in_test_set]


# load data
housing_data_pd = load_housing_data()
# --------------------------划分数据集--------------------------
# 1.手动划分数据集
# 先设置 index 用于后续计算 hash
# housing_with_id = housing_data_pd.reset_index() # adds an `index` column, 用于计算 hash
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")		# 用 index 计算 hash
# print(len(train_set), len(test_set))

# # 2.用 Scikit-Learn 划分数据集 (sklearn 还有其他函数可以划分数据集)
# from sklearn.model_selection import train_test_split
# # random_state 设置随机种子, 保证每次产生的随机是一样的
# train_set, test_set = train_test_split(housing_data_pd, test_size=0.2, random_state=42)
# print(len(train_set), len(test_set))


# 3.如果某个特征特别重要, 希望在划分数据集时测试集可以包含该特征的各个类别（让测试集更具代表性?）
# 添加 income_cat 属性代表 median_income 的类别, 这里分成 1~5 类
# category 1 ranges from 0 to 1.5 (i.e., less than $15,000), category 2 from 1.5 to 3, and so on
housing_data_pd["income_cat"] = pd.cut(housing_data_pd["median_income"], 
	bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# housing_data_pd["income_cat"].hist()
# plt.show()

# Now you are ready to do stratified sampling based on the income category.
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

strat_train_set = None
strat_test_set = None
for train_index, test_index in split.split(housing_data_pd, housing_data_pd["income_cat"]):
	strat_train_set = housing_data_pd.loc[train_index]
	strat_test_set = housing_data_pd.loc[test_index]

# 看到测试集中各类别的占比
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
# Now you should remove the income_cat attribute so the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
	set_.drop("income_cat", axis=1, inplace=True)

# Discover and Visualize the Data to Gain Insights
# create a copy so you can play with it without harming the training set
housing = strat_train_set.copy()
# # 可视化地理数据
# # 设 alpha 方便观察不同经纬度上点分布的密度
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 
# 	s=housing["population"]/100, label="population", figsize=(10, 7), 
# 	# c: 代表房价中位数, cmap=jet: 价格数字越大, 颜色越红, colorbar=True: 开启
# 	c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.legend()
# plt.show()
# # 可以根据图片增加特征, 以加强特征向量的表示
# # It will probably be useful to use a clustering algorithm to detect the main clusters, 
# # and add new features that measure the proximity to the cluster centers.

# standard correlation coefficient (also called Pearson’s r) 
corr_matrix = housing.corr()
# 例如： median_house_value 和其他属性的相关性
# print(corr_matrix["median_house_value"].sort_values(ascending=False))
# 相关系数范围：-1~1, 接近0意味着没有线性相关, 接近-1意味着负相关.(correlation coefficient 只测量线性相关)

# 另一种测量相关性的方法 TODO