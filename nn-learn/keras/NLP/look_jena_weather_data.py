import os


data_dir = 'data/jena_climate_2009_2016'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

# each line is a timestep: a record of a date and 14 weather-related values
# ['"Date Time"', '"p (mbar)"', '"T (degC)"', '"Tpot (K)"', '"Tdew (degC)"', '"rh (%)"', '"VPmax (mbar)"', '"VPact (mbar)"', '"VPdef (mbar)"', '"sh (g/kg)"', '"H2OC (mmol/mol)"', '"rho (g/m**3)"', '"wv (m/s)"', '"max. wv (m/s)"', '"wd (deg)"']
print('title')
print(header)
print('data')
print(lines[0])		# 数据以 , 分割
print('data size:', len(lines))


import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
	values = [float(x) for x in line.split(',')[1:]]
	float_data[i, :] = values


def plot_data():
	from matplotlib import pyplot as plt

	temp = float_data[:, 1] 	# 第一列是温度
	plt.figure(figsize=(10, 6))
	plt.subplot(1, 2, 1)
	plt.plot(range(len(temp)), temp)
	# the data is recorded every 10 minutes, you get 144 data points per day.
	plt.subplot(1, 2, 2)
	plt.plot(range(1440), temp[:1440])	# 只绘制前 144 天的温度数据(数据是每10分钟采集一次)
	plt.show()
plot_data()