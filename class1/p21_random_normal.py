import tensorflow as tf
a = tf.random.normal([2,2], mean=0, stddev=1)
print(a)
b = tf.random.truncated_normal([2,2], mean=0, stddev=1)
print(b)

