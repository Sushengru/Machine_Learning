import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.externals.six import StringIO


def clean(df):
    # 填充年龄的缺失值
    df['Age'] = df['Age'].fillna(df['Age'].mode()[0])

    # 上船港口的缺失值补充
    df['Embarked'].fillna('S', inplace=True)

    # print(df['Age'].mode(), '\n', df['Fare'].mode())

    # 将性别换成数字表示, 男1 女2
    # df['Sex'].loc[df['Sex'] == 'male', 'Sex'] = 1
    # df['Sex'].loc[df['Sex'] == 'female', 'Sex'] = 2
    df['Sex'].replace('male', 1, inplace=True)
    df['Sex'].replace('female', 2, inplace=True)

    # 港口换成数字表示， S1 C2 Q3
    # df['Embarked'].loc[df['Embarked'] == 'S', 'Embarked'] = 1
    # df['Embarked'].loc[df['Embarked'] == 'C', 'Embarked'] = 2
    # df['Embarked'].lco[df['Embarked'] == 'Q', 'Embarked'] = 3
    df['Embarked'].replace('S', 1, inplace=True)
    df['Embarked'].replace('C', 2, inplace=True)
    df['Embarked'].replace('Q', 3, inplace=True)

    columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", 'Survived']
    df = df[columns]

    return df


if __name__ == '__main__':
    path = 'train.csv'
    titanic = pd.read_csv(path)

    # 处理数据
    titanic = clean(titanic)
    # print(titanic.head(10), '\n', titanic.dtypes, '\n', titanic.describe(), '\n', titanic.info(), '\n', titanic.isnull().sum())

    # result 用于存储结果
    result = []

    # RF参数设置
    # 每次循环 增加最小子叶样本数量 5个
    sample_leaf_range = list(range(1, 150, 20))
    # 每次循环 增加决策树棵数 5棵
    n_estimators_range = list(range(50, 400, 50))

    # 随机生成测试集和训练集
    x = np.array(titanic[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]])
    y = np.array(titanic[['Survived']]).ravel()

    # 切分测试集、训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)

    # 循环最小子叶数和决策树棵树， 求出两者对于模型得分的影响情况
    # for leaf_num in sample_leaf_range:
    #     for estimators_num in n_estimators_range:
    #         # 建立模型
    #         rf_clf = RandomForestClassifier(min_samples_leaf=leaf_num, n_estimators=estimators_num, random_state=50,
    #                                         max_features='auto', max_depth=None, min_samples_split=5, )
    #         # 训练模型
    #         rf_clf.fit(x_train, y_train)
    #         # 得出测试集得分
    #         predict = rf_clf.score(x_test, y_test)
    #         # 用set保存对应数据
    #         result.append((leaf_num, estimators_num, predict))
    #         print(leaf_num, estimators_num, predict)
    #
    # print(max(result, key=lambda x: x[2]))

    """不使用循环， 使用GridSearchCV"""
    # # 建立模型
    # rf = RandomForestClassifier(max_depth=2, random_state=0, max_features=0.8)
    # # 需要调整的参数值列表
    # tuned_parameter = [{'min_samples_leaf': sample_leaf_range, 'n_estimators': n_estimators_range}]
    # # 用CV设置交叉验证
    # # clf = GridSearchCV(estimator=rf, param_grid=tuned_parameter, cv=5, n_jobs=1)
    # # 训练模型
    # clf.fit(x_train, y_train)
    # print('Best parameters:')
    # print(clf.best_params_)
    # print()


    """决策树"""
    tree_clf = tree.DecisionTreeClassifier(criterion='gini')
    clf = tree_clf.fit(x_train, y_train)

    # 打印分类器的参数情况
    print("clf:", str(clf))

    # from sklearn.externals.six import StringIO
    # import pydotplus
    #
    # # dot_data = StringIO()
    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # # graph.write_pdf('predict.pdf')

