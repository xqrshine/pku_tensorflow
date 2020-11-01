import tensorflow as tf
a = tf.zeros([2,3])
b = tf.ones(4)
c = tf.fill([2,2], 9)
print(a)
print(b)
print(c)
#
# tf.Tensor(
# [[0. 0. 0.]
#  [0. 0. 0.]], shape=(2, 3), dtype=float32)
# tf.Tensor([1. 1. 1. 1.], shape=(4,), dtype=float32)
# tf.Tensor(
# [[9 9]
#  [9 9]], shape=(2, 2), dtype=int32) 