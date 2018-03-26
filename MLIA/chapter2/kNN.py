# coding=utf-8
from numpy import *
import operator


# 创建数据集，确保每次输入相同的数据集
# 数据集如下，打相应tag A|B
# [[1.  1.1]  A
# [1.  1. ]   A
# [0.  0. ]   B
# [0.  0.1]]  B
def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# in_x（待归类的点坐标）
# data_set 数据集
# labels 坐标tag
# k 距离最小的k个点
def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    # (1)距离计算
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    #
    sorted_dist_indicies = distances.argsort()
    # (2)选择距离最小的k个点
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # (3)排序
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    group, label = create_data_set()
    print group
    print label
    x = classify0([1, 2], group, label, 3)
    print x
