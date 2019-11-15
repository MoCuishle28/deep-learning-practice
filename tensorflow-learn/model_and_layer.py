import tensorflow as tf


# 通过继承 tf.keras.Model 这个 Python 类来定义自己的模型
# 在继承类中，我们需要重写 __init__() 和 call(input) （模型调用）两个方法，同时也可以根据需要增加自定义的方法。
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)
        pass

    def call(self, input):
        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)
        # return output
        pass


X = tf.constant([[1.0, 2.0, 3.0], 
				[4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

# 自定义 Model 的方法写线性回归
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense: 全连接层
        self.dense = tf.keras.layers.Dense(
            units=1,			# 输出张量的维度
            activation=None,	# 激活函数, 包括 tf.nn.relu 、 tf.nn.tanh 和 tf.nn.sigmoid
            # use_bias ：是否加入偏置向量 bias, 默认为 True
            kernel_initializer=tf.zeros_initializer(),		# 默认为 tf.glorot_uniform_initializer
            bias_initializer=tf.zeros_initializer()			# 默认为 tf.glorot_uniform_initializer
        )

    def call(self, input):
        output = self.dense(input)
        return output


# 以下代码结构与前节类似
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)