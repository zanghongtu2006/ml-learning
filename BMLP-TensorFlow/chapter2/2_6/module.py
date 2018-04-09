#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-4-9 下午11:18
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

N = 210
K = 2

centers = [(2, -2), (-2, 1.5), (1.5, -2), (2, 1.5)]
data, features = datasets.make_blobs(n_samples=200, centers=centers, n_features=2,
                                     cluster_std=0.8, shuffle=False, random_state=42)

fig, ax = plt.subplots()
ax.scatter(np.asarray(data).transpose()[0],
           np.asarray(data).transpose()[1],
           marker='o', s=250)
plt.plot()

points = tf.Variable(data)
cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))
centroids = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [K, 2]))

fig1, ax1 = plt.subplots()
ax1.scatter(np.asarray(centers).transpose()[0],
            np.asarray(centers).transpose()[1],
            marker='o', s=250)
plt.show()
