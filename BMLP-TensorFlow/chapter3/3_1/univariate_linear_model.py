#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-4-16 下午10:39

import matplotlib.pyplot as plt
import numpy as np

trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.4 + 0.2

plt.figure()
plt.scatter(trX, trY)
plt.plot(trX, .2 + 2 * trX)
plt.show()
