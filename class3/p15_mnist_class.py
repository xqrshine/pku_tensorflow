import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_trian), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/225.0, x_test/225.0
print(x_test, y_test)


class MnistModel(Model):
    def __init__(self):
        """
        定义网络结构
        """
        super(MnistModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y


model = MnistModel()
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_trian, batch_size=1000, epochs=100, validation_data=(x_test, y_test), validation_freq=10)
model.summary()
