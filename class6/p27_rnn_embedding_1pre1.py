import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
import numpy as np
import matplotlib.pyplot as plt
import os


input_words = 'abcde'
words = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}  # 单词映射到id的字典
# words_one_hot = {0: [1., 0., 0., 0., 0.],
#                  1: [0., 1., 0., 0., 0.],
#                  2: [0., 0., 1., 0., 0.],
#                  3: [0., 0., 0., 1., 0.],
#                  4: [0., 0., 0., 0., 1.]
#                  }
# x_train = [words_one_hot[words['a']], words_one_hot[words['b']], words_one_hot[words['c']], words_one_hot[words['d']], words_one_hot[words['e']]]
x_train = [words['a'], words['b'], words['c'], words['d'], words['e']]
y_train = [words['b'], words['c'], words['d'], words['e'], words['a']]

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# x_train = np.reshape(x_train, (len(x_train), 1, 5))
# [送入样本数，循环核时间展开步数]
x_train = np.reshape(x_train, (len(x_train), 1))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    Embedding(5, 2),
    SimpleRNN(3),
    Dense(5, activation='softmax')
])
model.compile(optimizers=tf.keras.optimizers.Adam(lr=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path = './checkpoint/rnn_onehot_1pre1.ckpt'

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
plt.savefig('embedding_1pre1_acc.png')
plt.show()


plt.subplot(1, 2, 2)
plt.plot(loss, label='loss')
plt.title('train loss')
plt.savefig('embedding_1pre1_loss.png')
plt.show()


# 预测
preNumber = int(input("input the number of test alphabet:"))
for i in range(preNumber):
    alphbet = input('input the best alphbet:')
    alphbet = words[alphbet]
    alphbet = np.reshape(alphbet, (1, 1))
    result = model.predict([alphbet])
    pred = tf.argmax(result, axis=1)
    pred = int(pred)
    tf.print(alphbet, input_words[pred])