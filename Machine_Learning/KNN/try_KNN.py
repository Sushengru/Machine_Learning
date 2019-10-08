"""
    尝试自己实现KNN算法
    使用鸢尾花的数据
"""
import operator
from sklearn import datasets
import numpy as np


def norm_data(data_set, length):
    """
    ：desc
        KNN的简单实现过程
    :param
        data_set: 数据集( [[],[],...] )
    :return:
        normalized: 归一化处理的特征
    """
    # 取得每一列的最大、最小值
    max_dataset = data_set.max(0)
    min_dataset = data_set.min(0)

    # 最大值最小值做差
    diff_max_min = max_dataset - min_dataset
    # print(diff_max_min)

    # 生成最小值扩充矩阵
    sets = np.tile(min_dataset, (length, 1))
    # print(sets)

    # 1. 特征矩阵 - 最小值扩充矩阵
    diff = data_set - min_dataset

    # 2. 差值 / 列极差
    normalized = diff / diff_max_min

    # print(normalized)
    return normalized


def test_train(data_set, target):
    sample = np.random.randint(150, size=10)
    # print(sample)
    x_list = []
    y_list = []
    for i in range(len(sample)):
        x_list.append(data_set[sample[i], :])
        y_list.append(target[sample[i]])
    test_x = np.array(x_list)
    test_y = np.array(y_list)

    return test_x, test_y

    # data_set = np.array(data_set)
    # # print(data_set)
    # test_x = np.delete(data_set, train_x)
    # print(len(test_x))
    #
    # print(train_x)


def calculate_distance(test_x, data_set, target, k):
    """
    :desc
        KNN的分类函数
    :return:
    """
    for i in range(19):
        # 1. 计算 "中心点" 和数据集点的距离
        diff = test_x[i] - data_set

        # 2. 求平方和
        dis_sum = (diff ** 2).sum(axis=1)

        # 3. 开根号
        distance = dis_sum ** 0.5

        # 得到每个距离点的index
        sort_distince = distance.argsort()

        # 输入距离最近的 k 个点的类型值
        print(target[sort_distince[0:k]])

        # 根据类型值做统计
        num = np.bincount(target[sort_distince[0:k]])
        print(num)
        return num


if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # 归一化处理
    norm_data(x, len(x))

    # 随机选取n个点作为 "中心点"
    test_x, test_y = test_train(x, y)
    print(test_x)
    # 计算KNN距离，并归类结果
    num = calculate_distance(test_x, x, y, k=10)
