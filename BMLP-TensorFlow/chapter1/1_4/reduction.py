#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-3-31 下午11:18
import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.constant([[1, 2, 3],
                 [3, 2, 1],
                 [-1, -2, -3]])

boolean_tensor = tf.constant([[True, False, True],
                              [False, False, True],
                              [True, False, False]])

print(tf.reduce_prod(x, reduction_indices=1).eval())

print(tf.reduce_min(x, reduction_indices=1).eval())

print(tf.reduce_max(x, reduction_indices=1).eval())

print(tf.reduce_mean(x, reduction_indices=1).eval())

print(tf.reduce_all(boolean_tensor, reduction_indices=1).eval())

print(tf.reduce_any(boolean_tensor, reduction_indices=1).eval())
