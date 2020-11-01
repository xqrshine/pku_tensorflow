import tensorflow as tf
classes = 3
labels = tf.constant([2, 1, 0])
output = tf.one_hot(labels, depth=classes)
print(output)