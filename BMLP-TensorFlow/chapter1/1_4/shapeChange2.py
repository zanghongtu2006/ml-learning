#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-4-3 下午11:02
import tensorflow as tf

sess = tf.InteractiveSession()
t_matrix = tf.constant([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
t_array = tf.constant([1, 2, 3, 4, 9, 8, 6, 5])
t_array2 = tf.constant([2, 3, 4, 5, 6, 7, 8, 9])

# 切片，从[1,1]到[2,2]
print tf.slice(t_matrix, [1, 1], [2, 2]).eval()

# TODO: not finish
# 拆分
# print tf.split(0, 2, t_array)

print tf.tile([1, 2], [3]).eval()

# 填充
print tf.pad(t_matrix, [[0, 1], [2, 1]]).eval()

# print tf.concat(0, [t_array, t_array2]).eval()

# 该API已移除
# print tf.pack([t_array, t_array2]).eval()

print tf.reverse(t_matrix, [False, True]).eval()
print tf.reverse(t_array, [False]).eval()
