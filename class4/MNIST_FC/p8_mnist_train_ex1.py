import tensorflow as tf
from PIL import Image
import numpy as np
import os

train_path = './mnist_image_label/mnist_train_jpg_60000'
train_txt = './mnist_image_label/mnist_train_jpg_60000.txt'
x_train_savepath = './mnist_image_label/mnist_x_train.npy'
y_train_savepath = './mnist_image_label/mnist_y_train.npy'

test_path = './mnist_image_label/mnist_test_jpg_10000'
test_txt = './mnist_image_label/mnist_test_jpg_10000.txt'
x_test_savepath = './mnist_image_label/mnist_x_test.npy'
y_test_savepath = './mnist_image_label/mnist_y_test.npy'


def generated(path, txt):
    """
    读取原始图像，生成np.array数据格式和label
    :param path:
    :param txt:
    :return:
    """
    x, y_ = [], []
    f = open(txt, 'r')
    contents = f.readlines()
    for content in contents:
        value = content.split(' ')
        img_path = path + '/' + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))  # 灰度图像转为28*28宽灰度值的np.array格式
        img = img / 225.0
        x.append(img)
        y_.append(value[1])
    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x, y_


if os.path.exists(x_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(y_train_savepath) and os.path.exists(y_test_savepath):
    print('--------------------------加载数据------------------------------')
    x_train = np.load(x_train_savepath)
    x_test = np.load(x_test_savepath)
    y_train = np.load(y_train_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train, (len(x_train), 28, 28))
    x_test = np.reshape(x_test, (len(x_test), 28, 28))
else:
    print('生成数据：')
    x_train, y_train = generated(train_path, train_txt)
    x_test, y_test = generated(test_path, test_txt)
    print('数据保存到文件：')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(x_test_savepath, x_test_save)
    np.save(y_train_savepath, y_train)
    np.save(y_test_savepath, y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])  # 结构
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])  # 优化
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)  # 训练
model.summary()


