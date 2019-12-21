from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model

# 1个输入 3个输出
vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)

x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

# 3 个输出
# Note that the output layers are given names.
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

# 训练 3 个输出可能会需要 3 个不同的 Loss func, 例如：预测 age 是回归问题, 预测性别是分类问题
# keras 简答的做法是把不同的损失函数加在一起

# # 两种实现方法
# # list
# model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
# # dict
# model.compile(optimizer='rmsprop', 
# 	loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'})

# 由于不同损失函数计算出的结果数值范围差别很大, 还可以给不同的损失设定权重
# model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], 
# 	loss_weights=[0.25, 1., 10.])
model.compile(optimizer='rmsprop', 
	loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'}, 
	loss_weights={'age': 0.25, 'income': 1., 'gender': 10.})

# age_targets, income_targets, and gender_targets are assumed to be Numpy arrays.
# 还没有数据 posts, age_targets, income_targets, gender_targets

# model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)
model.fit(posts, {'age': age_targets, 'income': income_targets, 'gender': gender_targets}, 
	epochs=10, batch_size=64)