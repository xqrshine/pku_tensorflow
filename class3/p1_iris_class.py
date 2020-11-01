import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target
np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

class IrisModel(Model):
    def __init__(self):
        """
        定义网络结构块
        """
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        """
        调用网络结构块，实现前向传播
        :return:
        """
        y = self.d1(x)
        return y

model = IrisModel()
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metircs=['sparse_categorical_entropy']
              )  # metrics=['# 准确率']
model.fit(x_train, y_train, batch_size=30, epochs=1000, validation_split=0.2, validation_freq=100)
model.summary()
