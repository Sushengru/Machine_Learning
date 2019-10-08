"""
Author: yifan@netopstec.com

Create date: 2018-04-25

Description: 特征工程相关
    1. 无量纲化
        1.1 标准化
        1.2 区间缩放方法 —— 归一化

    2. 对定量特征二值化



Contain:
    1.
    2.

Update:

"""

from sklearn.datasets import load_iris



# 读取鸢尾花数据集
iris = load_iris()
x = iris.data
y = iris.target


# 1.1 无量纲化
# 1.1.1 标准化
# 标准化的前提是特征值服从正态分布，标准化后，其转换成标准正态分布。
from sklearn.preprocessing import StandardScaler
standar_x = StandardScaler().fit_transform(x)
print(standar_x[:10])

# 1.1.2 区间缩放法
# 将数据集缩放到[0, 1]区间
from sklearn.preprocessing import MinMaxScaler
MinMax_x = MinMaxScaler().fit_transform(x)
print(MinMax_x[:10])

# 1.1.3 L2正则化
from sklearn.preprocessing import Normalizer
normalize_x = Normalizer().fit_transform(x)
print(normalize_x[:10])

# 1.2 二值化
# 设定一个阈值，大于阈值的特征值赋值为1，反之赋值为0
from sklearn.preprocessing import Binarizer
binary_x = Binarizer(threshold=3).fit_transform(x)
print(binary_x[:10])

# 1.3 哑编码
from sklearn.preprocessing import OneHotEncoder
onehot_y = OneHotEncoder().fit_transform(y.reshape((-1, 1)))
print(y.reshape(-1, 1)[:10])
print(onehot_y[:10])

# 1.4 缺失值计算
# 参数strategy为缺失值填充方式，默认为mean
from numpy import vstack, array, nan
from sklearn.preprocessing import Imputer
vstack_x = vstack((array([nan,nan,nan,nan]), iris.data))
print(vstack_x[:10])
imput_x = Imputer().fit_transform(vstack_x)
print(imput_x[:10])

#
# 2. 特征筛选
# # 2.1 方差选择法
from sklearn.feature_selection import VarianceThreshold
var_x = VarianceThreshold(1).fit_transform(x)
print(var_x[:10])

# 2.2 卡方检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# 选择K个最好的特征
kb_chi2_x = SelectKBest(chi2, k=2).fit_transform(x, y)
print(kb_chi2_x[:10])

# 2.3 互信息法
