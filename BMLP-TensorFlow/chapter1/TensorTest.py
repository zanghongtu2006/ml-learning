# coding=utf-8
import numpy as np
import tensorflow as tf

# 1.1 初始化一个3阶张量
tens1 = tf.constant([[[1, 2], [2, 3]], [[3, 4], [5, 6]]])
print tens1

sess = tf.InteractiveSession()
# 取张量下的值
print sess.run(tens1)[1, 1, 0]

# 1.1.2 初始化随机张量
# np.random.rand(32) 32个随机数
x = tf.constant(np.random.rand(32).astype(np.float32))
y = tf.constant([1, 2, 3])

x_data = np.array([[1., 2., 3.], [3., 2., 6.]])
x1 = tf.convert_to_tensor(x_data, dtype=tf.float32)

print x1

b = tf.Variable(tf.zeros([1000]))
print b
