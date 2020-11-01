import tensorflow as tf
import numpy as np

SEED=23455
rdm = np.random.RandomState(seed=SEED)
x = rdm.rand(32, 2)
y_ = np.array([[x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1, x2) in x])
x = tf.cast(x, dtype=tf.float32)

w = tf.Variable(tf.random.normal([2, 1], stddev=0.25, seed=1))

epochs = 1000
lr = 0.01

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w)

        loss_mse = tf.reduce_mean(tf.square(y_ - y), axis=0)  # y_ture, y_pre在指定axis做均值计算，默认是0

        # loss_mse = tf.keras.losses.MSE(y_, y)
        # print(loss_mse)
    grad = tape.gradient(loss_mse, w)
    w.assign_sub(lr*grad)
    if epoch % 200 == 0:
        print('epoch:{} \n w:{}'.format(epoch, w.numpy()))
        print('loss_mse:{}'.format(loss_mse))