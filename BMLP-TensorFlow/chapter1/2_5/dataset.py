#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-4-9 下午10:29

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

data, features = datasets.make_blobs(n_samples=100, n_features=2, centers=3,
                                     cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True,
                                     random_state=None)

data1, features1 = datasets.make_circles(n_samples=100, shuffle=True, noise=None,
                                         random_state=None, factor=0.8)

fig, ax = plt.subplots()
# ax.scatter(np.asarray(data).transpose()[0],
#            np.asarray(data).transpose()[1],
#            marker='o', s=25)
# plt.show()

ax.scatter(np.asarray(data1).transpose()[0],
           np.asarray(data1).transpose()[1],
           marker='o', s=25)
plt.show()
