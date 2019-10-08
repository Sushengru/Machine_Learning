"""
    sklearn 实现KNN测试
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler


if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    print(x)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # StandardScaler 标准化
    s = StandardScaler()
    s_x = s.fit_transform(x)
    print(s_x)
    ss_x_test = s.fit_transform(x_test)
    print(ss_x_test)

    # normalize 标准化
    n_x_text = normalize(x_test)
    print(n_x_text)

    # 使用KNN
    knc = KNeighborsClassifier(algorithm='auto')
    knc.fit(x_train, y_train)

    # 使用模型自带的评估函数进行准确性评估
    point = knc.score(x_test, y_test)
    print(point)

