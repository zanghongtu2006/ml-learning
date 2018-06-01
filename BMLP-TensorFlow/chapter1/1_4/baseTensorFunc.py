#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-3-31 下午10:47

import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.constant([[2, 5, 3, -5],
                 [0, 3, -2, 5],
                 [4, 3, 5, 3],
                 [6, 1, 4, 0]])
y = tf.constant([[4, -7, 4, -3, 4],
                 [6, 4, -7, 4, 7],
                 [2, 3, 2, 1, 4],
                 [1, 5, 5, 5, 2]])
floatx = tf.constant([[2., 5., 3., -5.],
                      [0., 3., -2., 5.],
                      [4., 3., 5., 3.],
                      [6., 1., 4., 0.]])
print(tf.transpose(x).eval())  # 矩阵倒置
print(tf.matmul(x, y).eval())  # 矩阵乘法
print(tf.matrix_determinant(floatx).eval())
print(tf.matrix_inverse(floatx).eval())  # 矩阵逆运算
print(tf.matrix_solve(floatx, [[1], [1], [1], [1]]).eval())
