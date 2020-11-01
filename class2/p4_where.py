import tensorflow as tf
a = tf.constant([1, 2, 3, 4])
b = tf.constant([5, 6, 7, 8])
print(tf.where(tf.greater(a, b), a, b))
