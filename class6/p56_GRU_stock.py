import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dropout, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os


maotai = pd.read_csv('./SH600519.csv')
# 开盘价作为可用数据
train_set = maotai.iloc[0:2428-300, 2:3].values
test_set = maotai.iloc[2428-300:, 2:3].values
# 归一化
sc = MinMaxScaler(feature_range=(0, 1))
train_set = sc.fit_transform(train_set)
test_set = sc.transform(test_set)

x_train = []
x_test = []
y_train = []
y_test = []
# 构造训练集和测试集, 前60个数据作为训练数据，第61个数据作为label
for i in range(len(train_set)-60):
    x_train.append(train_set[i:i+60, 0])
    y_train.append(train_set[i+60, 0])

for i in range(len(test_set)-60):
    x_test.append(test_set[i:i+60, 0])
    y_test.append(test_set[i+60, 0])
# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 将训练集和测试集由list转为numpy格式
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)
# x_train, x_test需要符合RNN的输入要求：[送入样本数，循环核时间展开步数，每个时间步输入特征个数]
# 第一次错在了输入格式，本例中没有用词向量表示，少加了‘每个时间步输入特征个数’
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

# 构建神经网络
model = tf.keras.Sequential([
    GRU(80, return_sequences=True),
    Dropout(0.2),
    GRU(100),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),
              loss='mean_squared_error',
              Metrics=['sparse_categorical_accuracy'])
checkpoint_filepath = './checkpoint/gru_stock.ckpt'
if os.path.exists(checkpoint_filepath + '.index'):
    model.load_weights(checkpoint_filepath)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
# history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), validation_freq=1,
#                     callbacks=[model_checkpoint_callback])
history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[model_checkpoint_callback])
model.summary()

# 参数提取
file = open('./gru_stock_weight.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

# 准确率和损失可视化
val_loss = history.history['val_loss']
loss = history.history['loss']
plt.subplot(1, 2, 1)
plt.plot(loss, label='train_loss')
plt.savefig('./gru_stock_train_loss')
plt.show()
plt.subplot(1, 2, 2)
plt.plot(val_loss, label='val_loss')
plt.savefig('gru_stock_val_loss')
plt.show()

# 对测试集进行预测
predict_stock_price = model.predict(x_test)
# 对原始数据反归一化
real_stock_price = sc.inverse_transform(test_set[60:])
# 对预测数据反归一化
predict_stock_price = sc.inverse_transform(predict_stock_price)
# 均方误差
mse = mean_squared_error(predict_stock_price, real_stock_price)
# 均方根误差
smse = math.sqrt(mse)
# 均绝对值误差
mae = mean_absolute_error(predict_stock_price, real_stock_price)
print('{}\n{}\n{}'.format(mse, smse, mae))
