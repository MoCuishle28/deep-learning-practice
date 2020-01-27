import os
import tarfile
import pandas as pd
from six.moves import urllib


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("../datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	if not os.path.isdir(housing_path):
		os.makedirs(housing_path)
	tgz_path = os.path.join(housing_path, "housing.tgz")
	urllib.request.urlretrieve(housing_url, tgz_path)
	housing_tgz = tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, "housing.csv")
	return pd.read_csv(csv_path)


# fetch_housing_data()   # download
data_pd = load_housing_data()		# 20640
print(data_pd.head())
print(data_pd.info())

print('---value_counts---')
print(data_pd['ocean_proximity'].value_counts())

print('---describe---')
# Note that the null values are ignored 
# (so, for example, count of total_bedrooms is 20,433, not 20,640)
print(data_pd.describe())


# plot
import matplotlib.pyplot as plt
data_pd.hist(bins=50, figsize=(20, 15))
plt.show()