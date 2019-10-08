import operator as op
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator


def createDataSet():
    pass



def fiel_2_matrix(filename):
    """
    :desc
        导入训练数据集
    :param
        filename: 数据文件路径
    :return:
        数据矩阵 returnMat, 和对应的类别 classLabelVector
    """

    fr = open(filename)

    # 获得文件中的数据行的行数
    line_num = len(fr.readlines())
    print(line_num)

    # 生成对应的空矩阵
    # 例如：zeros(line_num, 3) 就是生成一个2*3的矩阵，各个位置上全是0
    return_mat = np.zeros((line_num, 3))
    class_label_vetor = []

    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # 移除字符串首尾制定的字符生成新的字符串
        line = line.strip()

        # 以 '\t' 切割字符串
        list_from_line = line.split('\t')

        # 每列的属性数据 添加到矩阵行中, 按行不断替换
        return_mat[index, :] = list_from_line[0:3]

        # 每列的类别数据，就是label的标签数据
        class_label_vetor.append(int(list_from_line[-1]))

        index += 1

    # 返回数据矩阵return_mat 和对应的类别 class_label_vetor
    return return_mat, class_label_vetor


def show_map(return_mat, class_label_vetor):
    """
    : desc
        使用Matplotlib画二维散点图
    :param
        return_mat:
    :param
        class_label_vetor:
    :return:
        没有return， 只显示散点图形
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(return_mat[:, 1], return_mat[:, 2],
               15.0 * np.array(class_label_vetor), 15.0 * np.array(class_label_vetor))
    plt.show()


def auto_norm(dataset):
    """
    Desc:
        归一化特征值，消除特征之间量级不同导致的影响
    :param
        dataset: 数据集
    :return:
        归一化后的数据集 norm_dataset, ranges 和 min_vals 即 归一化矩阵, 范围和最小值
    归一化公式：
        Y = (X - Xmin) / (Xmax - Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。
        该函数可以自动将数字特征值转化为 0~1 的区间。
    """
    # 计算每个属性（每列）的最大值、最小值、范围
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    # 极差
    ranges = max_vals - min_vals

    norm_dataset = np.zeros(np.shape(dataset))
    # m 为数据集的第一维的长度，相当于行数
    m = dataset.shape[0]

    # 生成与最小值的差组成的矩阵
    norm_dataset = dataset - np.tile(min_vals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    norm_dataset = norm_dataset / np.tile(ranges, (m, 1))

    return norm_dataset, ranges, min_vals


def classify0(inx, dataset, labels, k):
    """
    Desc:
        KNN的分类函数
    :param
        inx: 用于分类的输入向量/测试数据
        dataset: 训练数据集的feature
        labels: 训练数据集的labels
        k: 选择最近邻的数目
    :return:
        sorted_class_count[0][0]: 输入向量的预测分类 labels


    """
    # ----------------实现 classify0 方法的第一种方式----------------
    # 1. 距离计算
    dataset_size = dataset.shape[0]

    # tile 生成和训练样本对应的矩阵，并与训练样本求差
    diff_mat = np.tile(inx, (dataset_size, 1)) - dataset

    # 欧式距离计算
    # 取平方
    sq_diff_mat = diff_mat ** 2
    # 将矩阵的每一行相加
    sq_distance = sq_diff_mat.sum(axis=1)
    # 对结果开根号, distance.shape = [m,1]
    distance = sq_distance ** 0.5

    # 根据距离排序从小到大的排序，返回对应的索引位置
    sorted_dist_indicies = distance.argsort()
    print(sorted_dist_indicies[0])

    # 2. 选择距离最小的k个点
    class_count = {}
    for i in range(k):
        # 找到该样本的类型,即Y值
        vote_i_label = labels[sorted_dist_indicies[i]]

        # 在字典中将该类型加一
        # 例如， y =[1,1,2], 则最终 class_count = {1: 2, 2: 1}
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    # 3. 排序并返回出现最多的那个类型
    sorted_class_count = sorted(class_count.items(), key=op.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def dating_class_test():
    """
    Desc:
        对约会网站的测试方法
    :param:
        None
    :return:
        错误数
    """
    # 设置测试数据的一个比例 (训练数据集比例 = 1 - ho_ratio)
    ho_ratio = 0.1

    # 从文本中加载数据
    path = 'F:\\Python_work\\Python_Study\\MachineLearning\\KNN\\datingTestSet2.txt'
    dating_data_mat, dating_labels = fiel_2_matrix(path)
    # 散点图展示
    show_map(dating_data_mat, dating_labels)
    # 归一化数据
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)

    # m为测试集行数
    m = norm_mat.shape[0]
    # 设置测试的样本数量 num_test_vecs:m 表示训练样本的数量
    num_test_vecs = int(m * ho_ratio)
    print('num_test_vecs:', num_test_vecs)

    error_count = 0

    for i in range(num_test_vecs):
        # 对数据测试
        class_ifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (class_ifier_result, dating_labels[i]))
        if (class_ifier_result != dating_labels[i]):
            error_count += 1
    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))
    print(error_count)


def class_dify_person():
    # Y的取值类型
    result_list = ['not at all', 'in samll doses', 'in large doses']

    # 输入参数
    percent_tats = float(input("percentage of time spent playing video games ? "))
    ff_miles = float(input("frequent filer miles earned per year ? "))
    ice_cream = float(input("liters of ice cream consumed per year ? "))

    # 读取数据
    dating_data_mat, dating_labels = fiel_2_matrix('datingTestSet2.txt')
    # 散点图展示
    show_map(dating_data_mat, dating_labels)
    # 归一化样本
    normat, ranges, min_vals = auto_norm(dating_data_mat)

    in_arr = np.array([ff_miles, percent_tats, ice_cream])

    class_ifier_result = classify0((in_arr - min_vals) / ranges, normat,dating_data_mat, 3)

    print("You will probably like this person: ", result_list[class_ifier_result - 1])


if __name__ == '__main__':
    # # 获取数据地址读取数据，并返回returnMat 和 classLabelVector
    # file_path = 'F:\\Python_work\\Python_Study\\MachineLearning\\KNN\\datingTestSet2.txt'
    # print(file_path)
    # x, y = fiel_2_matrix(file_path)
    # print(x)
    # print(y)
    #
    # show_map(x, y)
    dating_class_test()