import pickle


# 是 20M 的数据集的 embedding
path = 'ml_embedding_pca128.pkl'

with open('../data/'+path, 'rb') as f:
	data = pickle.load(f)

# 1~131262
print(min(data.keys()), max(data.keys()))