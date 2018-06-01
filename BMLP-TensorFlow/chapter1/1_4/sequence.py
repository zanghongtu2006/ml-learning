#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-4-2 下午11:18
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.constant([[2, 5, 3, -5],
                 [0, 3, -2, 5],
                 [4, 3, 5, 3],
                 [6, 1, 4, 0]])
listx = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
listy = tf.constant([4, 5, 8, 9])
boolx = tf.constant([[True, False], [False, True]])

# 返回input最小值的索引index
print(tf.argmin(x, 1).eval())

print(tf.argmax(x, 1).eval())

# 原书中API，该版本已无此API
# print tf.listdiff(listx, listy)[0].eval()

# 值为true的坐标
print(tf.where(boolx).eval())

# 去重
print(tf.unique(listx)[0].eval())
