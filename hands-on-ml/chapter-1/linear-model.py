import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


def prepare_country_stats(oecd_bli, gdp_per_capita):
	oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
	oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
	gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
	gdp_per_capita.set_index("Country", inplace=True)
	full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
								  left_index=True, right_index=True)
	full_country_stats.sort_values(by="GDP per capita", inplace=True)
	remove_indices = [0, 1, 6, 8, 33, 34, 35]
	keep_indices = list(set(range(36)) - set(remove_indices))
	return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


base_dir = '../datasets/'

# Load the data
oecd_bli = pd.read_csv(base_dir + "oecd_bli_2015.csv", thousands=',')
# print(oecd_bli)
'''
index 	LOCATION      Country 	INDICATOR  	...  Value 	Flag Codes          Flags
0         AUS       Australia   HO_BASE  	...   1.10          E  			Estimated value
1         AUT         Austria   HO_BASE  	...   1.00        NaN              NaN
...       ...             ...       ...  	...    ...        ...              ...
3287      EST         Estonia   WL_TNOW  	...  14.43        NaN              NaN
'''
gdp_per_capita = pd.read_csv(base_dir + "gdp_per_capita.csv",thousands=',',delimiter='\t', 
	encoding='latin1', na_values="n/a")
# print(gdp_per_capita)
'''
										   Country  	... 		Estimates Start After
0                                          Afghanistan  ...                2013.0
1                                              Albania  ...                2010.0
..                                                 ...  ...                   ...
189  International Monetary Fund, World Economic Ou...  ...                   NaN
'''

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Select a linear model
model = sklearn.linear_model.LinearRegression()
# 也可以选择 KNN 算法, 只需要将上面代码替换为以下代码
# import sklearn.neighbors
# model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus' GDP per capita
predict_y = model.predict(X_new)
print(predict_y) # outputs [[ 5.96242338]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.scatter(22587, predict_y, color = 'red')
# 添加注释, xytext-> 注解放置位置
plt.annotate("Cyprus' satisfaction", xy=(22587, predict_y), xytext=(22587, predict_y - 1), arrowprops=dict(arrowstyle='->'))
plt.show()