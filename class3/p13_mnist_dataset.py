import tensorflow as tf
# from sklearn import datasets  # sklearn没有mnist数据集
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_trian), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/225.0, x_test/225.0
print(x_test, y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_trian, batch_size=1000, epochs=500, validation_data=(x_test, y_test), validation_freq=100)
model.summary()

