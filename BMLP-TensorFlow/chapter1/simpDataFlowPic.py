#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-3-31 下午10:08
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    # import tensorflow as tf

    sess = tf.Session()
    W_m = tf.Variable(tf.zeros([10, 5]))
    x_v = tf.placeholder(tf.float32, [None, 10])
    result = tf.matmul(x_v, W_m)
    print g.as_graph_def()
