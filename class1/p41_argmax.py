import tensorflow as tf
import numpy as np

test = np.array([[1,3,2], [4,5,6], [7,8,9]])
print(tf.argmax(test, axis=1))