import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dot.csv')
x_data = np.array(data[['x1', 'x2']])
y_data = np.array(data['y_c'])
# print(x_data.shape)
# print(y_data.shape)

Y_c = [['red' if y else 'blue'] for y in y_data]
#对x y数据类型转换，防止训练的时候报错
x_data = tf.cast(x_data, tf.float32)
y_data = tf.cast(y_data, tf.float32)
# from_tensor_slices 生成标签对
train_db = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(32)
# 生成训练参数，输入层2个特征，一个隐藏层（2，11），输出层2个特征
# 两层可训练参数
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.1, shape=[1]))
w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.1, shape=[1]))

# 训练部分，梯度下降学习参数
epochs = 1000
lr = 0.02
for epoch in range(epochs):
    for step, (x1, x2) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_data, w1) + b1
            h1 = tf.nn.relu(h1)  # relu激活函数激活隐藏层
            y = tf.matmul(h1, w2) + b2
            y = tf.nn.softmax(y)  # softmax激活函数归一化
            loss = tf.reduce_mean(tf.square(y - y_data))
        # 计算loss对每个参数的梯度
        weight = [w1, b1, w2, b2]
        grad = tape.gradient(loss, weight)
        # 梯度更新
        w1.assign_sub(lr * grad[0])
        b2.assign_sub(lr * grad[1])
        w2.assign_sub(lr * grad[2])
        b2.assign_sub(lr * grad[3])
    if epoch % 100 == 0:
        print('epoch:{}\n weight:{}{}{}{}\n loss:{}'.format(epoch, w1, b1, w2, b2, loss))

# 预测部分
print('---------------------predict--------------------')
xx, yy = np.mgrid[-3:3:0.1, -3:3:0.1]  # 网格数据
test = np.c_[xx.ravel(), yy.ravel()]
test = tf.cast(test, dtype=tf.float32)
# 将数据喂入神经网络预测结果
prob = []
for x_test in test:
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y_test = tf.matmul(h1, w2) + b2
    prob.append(y_test)
# 绘图
x1 = x_data[:, 0]
x2 = x_data[:, 1]
plt.scatter(x1, x2, color=np.squeeze(Y_c))
prob = np.array(prob).reshape(xx.shape)
plt.contour(xx, yy, prob, levels=[.5])
plt.show()










