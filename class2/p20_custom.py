import tensorflow as tf
import numpy as np

SEED=23455
rdm = np.random.RandomState(seed=SEED)
x = rdm.rand(32, 2)
y_ = np.array([[x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1, x2) in x])
x = tf.cast(x, dtype=tf.float32)

w = tf.Variable(tf.random.normal([2, 1], stddev=0.25, seed=1))

epochs = 100000
lr = 0.002
COST = 1
PROFIT = 99
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w)
        loss_custom = tf.reduce_sum(tf.where(tf.greater(y, y_), COST * (y - y_), PROFIT * (y_ - y)))
        # print(loss_mse)
    grad = tape.gradient(loss_custom, w)
    w.assign_sub(lr*grad)
    if epoch % 1000 == 0:
        print('epoch:{} \n w:{}'.format(epoch, w.numpy()))
        print('loss_mse:{}'.format(loss_custom))
