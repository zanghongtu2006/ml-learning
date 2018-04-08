#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-4-8 下午11:38
# 随机生成100个点
import matplotlib.pyplot as plt
import tensorflow as tf

with tf.Session() as session:
    fig, ax = plt.subplots()
    ax.plot(tf.random_normal([100]).eval(),
            tf.random_normal([100]).eval(),
            'o')
    ax.set_title('Sample random plot for TensorFlow')
    plt.savefig("result.png")
