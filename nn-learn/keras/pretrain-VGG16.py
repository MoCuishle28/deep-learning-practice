# from tensorflow.keras.applications import VGG16
# 下载不了 https://github.com/fchollet/deep-learning-models/releases

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))