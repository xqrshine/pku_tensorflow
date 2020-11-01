import tensorflow as tf
x1 = tf.constant([[1,2],[3,4]], dtype=tf.float64)
print(x1)
x2 = tf.cast(x1, dtype=tf.int64)
print(x2)
print(tf.reduce_min(x1), tf.reduce_max(x1))