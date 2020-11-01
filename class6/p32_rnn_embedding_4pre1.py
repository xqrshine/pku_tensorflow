import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
import numpy as np
import matplotlib.pyplot as plt
import os


input_words = 'abcdefghijklmnopqrstuvwxyz'
# 单词映射到id的字典
words = {}
train_set_scaled = []
for i in range(26):
    words[input_words[i]] = i
    train_set_scaled.append(i)
# words = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
# words_one_hot = {0: [1., 0., 0., 0., 0.],
#                  1: [0., 1., 0., 0., 0.],
#                  2: [0., 0., 1., 0., 0.],
#                  3: [0., 0., 0., 1., 0.],
#                  4: [0., 0., 0., 0., 1.]
#                  }
# 输入4个单词，预测下一个单词
x_train = []
y_train = []

for i in range(22):
    x_train.append(train_set_scaled[i:i+4])
    y_train.append(train_set_scaled[i])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# [送入样本数，循环核时间展开数]
x_train = np.reshape(x_train, (len(x_train), 4))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    Embedding(26, 2),
    SimpleRNN(5),
    Dense(26, activation='softmax')
])
model.compile(optimizers=tf.keras.optimizers.Adam(lr=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path = './checkpoint/rnn_embebdding_4pre1.ckpt'

if os.path.exists(checkpoint_save_path + '.index'):
    print('-----------------load model-----------------')
    model.load_weights(checkpoint_save_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')
history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])



model.summary()

file = open('./weight.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='accuracy')
plt.title('train accuracy')
plt.savefig('onehot_4pre1_acc.png')
plt.show()


plt.subplot(1, 2, 2)
plt.plot(loss, label='loss')
plt.title('train loss')
plt.savefig('onehot_4pre1_loss.png')
plt.show()


# 预测
preNumber = int(input("input the number of test alphabet:"))
for i in range(preNumber):
    alphbet = input('input the best alphbet:')
    # 预测数据格式
    # alphbet = [words_one_hot[words[alphbet[0]]], words_one_hot[words[alphbet[1]]], words_one_hot[words[alphbet[2]]],
    #            words_one_hot[words[alphbet[3]]]]
    alphbet = [words[i] for i in alphbet]
    alphbet = np.reshape(alphbet, (1, 4))
    result = model.predict([alphbet])
    pred = tf.argmax(result, axis=1)
    pred = int(pred)
    tf.print(alphbet, input_words[pred])