import tensorflow as tf
x1 = tf.constant([[1,2],[3,4]], dtype=tf.float64)
print(x1)
print(tf.reduce_mean(x1, axis=0), tf.reduce_mean(x1, axis=1))
print(tf.reduce_sum(x1, axis=0), tf.reduce_sum(x1, axis=1))