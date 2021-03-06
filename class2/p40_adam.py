import tensorflow as tf
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
# print(x_data)
# print(y_data)

np.random.seed(116)  # 使用相同的seed，使输入特征/标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30]
x_test = x_data[-30:]
y_train = y_data[:-30]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义训练参数
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

train_loss_results = []
acc_results = []
lr = 0.1
loss_all = 0
epochs = 200
# 一阶动量
m_w, m_b = 0, 0
beta_1 = 0.9
# 二阶动量
v_w, v_b = 0, 0
EPSILON = 1e-6
beta_2 = 0.999
for epoch in range(epochs):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            # y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y - y_))
            loss_all += loss.numpy()
        # 计算梯度
        grad = tape.gradient(loss, [w1, b1])
        m_w = beta_1 * m_w + (1 - beta_1) * grad[0]
        m_b = beta_1 * m_b + (1 - beta_1) * grad[1]
        v_w = beta_2 * v_w + (1 - beta_2) * tf.square(grad[0])
        v_b = beta_2 * v_b + (1 - beta_2) * tf.square(grad[1])
        m_w_correct = m_w / (1 - tf.pow(beta_1, (step + 1) * (epoch + 1)))
        m_b_correct = m_b / (1 - tf.pow(beta_1, (step + 1) * (epoch + 1)))
        v_w_correct = v_w / (1 - tf.pow(beta_2, (step + 1) * (epoch + 1)))
        v_b_correct = v_b / (1 - tf.pow(beta_2, (step + 1) * (epoch + 1)))
        # 实现梯度更新
        print(m_w_correct)
        print(v_w_correct)
        w1.assign_sub(lr*m_w_correct/tf.sqrt(v_w_correct + EPSILON))
        print(m_b_correct)
        print(v_b_correct)
        b1.assign_sub(lr*m_b_correct/tf.sqrt(v_b_correct + EPSILON))
    print("epoch:{} loss:{}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all = 0  # loss_all 归零，为记录下一个epoch的loss做准备

    total_correct = 0
    total_number = 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += correct
        total_number += x_test.shape[0]
    acc = total_correct/total_number
    print('acc:{}'.format(acc))
    acc_results.append(acc.numpy())
print(acc_results)

# 绘制loss曲线
plt.title('Loss Function Curve')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label='$Loss$')
plt.legend()
plt.show()

# 绘制acc曲线
plt.title('accuracy Curve')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(acc_results, label='$Accuracy$')
plt.legend()
plt.show()

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(acc_results, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()