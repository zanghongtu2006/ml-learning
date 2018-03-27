# coding=utf-8

import matplotlib.pyplot as plt
from numpy import *


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
    class_count = {}  # in_x和所有已知点的距离
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # (3)排序
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


# 读文件
# 文件格式： 40920	8.326976	0.953952	3
#          飞行里程    游戏时间    冰淇淋数     得分
def file_2_matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读出所有行
    array_o_lines = fr.readlines()
    # 读出行数
    number_of_lines = len(array_o_lines)
    # 初始化一个矩阵和一个得分集合，文件的前3位存到矩阵中，最后一位存到结果集
    # zeros() 创建一个2维数组，长度是行数，列是3
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    # index是行索引
    index = 0
    for line in array_o_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        # print list_from_line
        # 把每行split结果的前3位赋值给矩阵的一行，最后一位赋值给class_label_vector
        return_mat[index, :] = list_from_line[0:3]
        # print return_mat
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def showData(mat, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 输出矩阵的第2,3列数据
    # ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])
    # 输出矩阵的第2,列数据，使用色彩，尺寸表示
    # 横轴是飞行里程数(0),纵轴是游戏时间(1),彩色点是得分
    ax.scatter(mat[:, 0], mat[:, 1],
               15.0 * array(label), 15.0 * array(label))
    plt.show()


def autoNorm(data_set):
    # 每列最小值
    min_vals = data_set.min(0)
    # 每列最大值
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def datingClassTest():
    ho_ratio = 0.10
    dating_data_mat, dating_data_label = file_2_matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                      dating_data_label[num_test_vecs:m], 3)
        print "the classifier came back with:%d, the real answer is %d" \
              % (classifier_result, dating_data_label[i])
        if classifier_result != dating_data_label[i]:
            error_count += 0
    print "the total error rate is: %f" % (error_count / float(num_test_vecs))


if __name__ == '__main__':
    dating_data_mat, dating_data_label = file_2_matrix('datingTestSet2.txt')
    norm_dating_data_mat, norm_ranges, norm_min_vals = autoNorm(dating_data_mat)
    print(dating_data_mat)
    print(norm_dating_data_mat)
    showData(norm_dating_data_mat, dating_data_label)
