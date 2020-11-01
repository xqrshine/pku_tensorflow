import tensorflow as tf
with tf.GradientTape() as gt:
    w = tf.Variable(tf.constant(3.0))
    loss = tf.pow(w, 2)
grad = gt.gradient(loss, w)
print(grad)