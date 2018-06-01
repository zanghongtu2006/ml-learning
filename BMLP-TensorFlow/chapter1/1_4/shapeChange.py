#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-4-3 下午10:47
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.constant([[2, 5, 3, -5],
                 [0, 3, -2, 5],
                 [4, 3, 5, 3],
                 [6, 1, 4, 0]])

# 输出矩阵形状
print(tf.shape(x).eval())

# 输出矩阵元素个数
print(tf.size(x).eval())

# 输出矩阵维度
print(tf.rank(x).eval())

# 重新排列矩阵
print(tf.reshape(x, [8, 2]).eval())

print(tf.squeeze(x).eval())

# 插入维度1进入一个tensor中
print(tf.expand_dims(x, 1).eval())
