#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-4-9 下午11:18
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

N = 200
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
# plt.show()

# 损失函数描述和优化循环
rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, 2])
sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
                            reduction_indices=2)
best_centroids = tf.argmin(sum_squares, 1)
print best_centroids

# 停止条件
did_assignments_chage = tf.reduce_any(tf.not_equal(best_centroids,
                                                   cluster_assignments))


def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count


means = bucket_mean(points, best_centroids, K)

with tf.control_dependencies([did_assignments_chage]):
    do_updates = tf.group(centroids.assign(means),
                          cluster_assignments.assign(best_centroids))

print best_centroids
plt.show()
