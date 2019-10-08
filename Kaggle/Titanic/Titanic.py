import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier


def fill_data(data):
    # 众数填充 港口和船票费用
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].mode()[0], inplace=True)

    # 用上下共2个标准差的距离填充缺失age
    fill_age(data)

    return data


def fill_age(data):
    age_mean = data['Age'].mean()
    age_std = data['Age'].std()
    age_null_count = data['Age'].isnull().sum()

    age_random_fill = np.random.randint(age_mean - age_std, age_mean + age_std, size=age_null_count)

    data['Age'][np.isnan(data['Age'])] = age_random_fill

    data['Age'] = data['Age'].astype(int)


def add_columns(data):
    # 添加新特征
    # family_size： 家庭人数
    # is_alone ： 是否单独一人

    data['Family_size'] = data['SibSp'] + data['Parch'] + 1
    data['Is_alone'] = 0
    data.loc[data['Family_size'] > 1, 'Is_alone'] = 1

    data['Has_cabin'] = data['Cabin'].apply(lambda x: 1 if type(x) == float else 0)

    return data


def get_prefix(name):
    prefix = re.search(' ([a-zA-Z]+)\.', name)
    if prefix:
        return prefix.group(1)
    return


def clean_data(data):
    # 提取名称前缀
    data['Prefix'] = data['Name'].apply(get_prefix)

    # 简化归类特征值
    data['Prefix'] = data['Prefix'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Prefix'] = data['Prefix'].replace('Mlle', 'Miss')
    data['Prefix'] = data['Prefix'].replace('Ms', 'Miss')
    data['Prefix'] = data['Prefix'].replace('Mme', 'Mrs')
    data['Prefix'].fillna('Mr', inplace=True)
    return data


def discrete_data(data):
    # 标签化和离散化

    data['Gender'] = data['Sex'].map({'male': 1, 'female': 2}).astype(int)
    data['Port'] = data['Embarked'].map({'S': 1, 'Q': 2, 'C': 3}).astype(int)

    # Age
    data.loc[data['Age'] <= 16, 'Age_range'] = 1
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age_range'] = 2
    data.loc[(data['Age'] > 32) & (data['Age'] <= 45), 'Age_range'] = 3
    data.loc[(data['Age'] > 45) & (data['Age'] <= 60), 'Age_range'] = 4
    data.loc[data['Age'] > 60, 'Age_range'] = 5

    # Fare
    fare_list = [1, 2, 3, 4, 5]
    data['Fare_range'] = pd.cut(data['Fare'], bins=5, labels=fare_list)
    data['Fare_range'] = data['Fare_range'].astype('float')

    # Title
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Prefix'] = data['Prefix'].map(title_mapping)

    return data


if __name__ == '__main__':

    train_path = r'F:\Python_work\Python_Study\Kaggle\train.csv'
    test_path = r'F:\Python_work\Python_Study\Kaggle\test.csv'

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 1. 清洗
    for dataset in [train, test]:
        print(dataset.info(), dataset.describe(), dataset.dtypes, dataset.head(12))

        # 1.1 填充缺失值
        dataset = fill_data(dataset)

        # 1.2 生成新的特征字段
        dataset = add_columns(dataset)

        # 1.3 清洗特征
        dataset = clean_data(dataset)

        # 1.4 离散化、数字化
        dataset = discrete_data(dataset)

        print(dataset.info(), dataset.describe())

    # 1.5 选出需要使用的特征
    # train.to_csv('train_data.csv')

    # 年龄， 性别， 港口， 票价， 是否有船舱， 尊称前缀， 家庭大小， 是否一人，船舱级别
    select_list = ['Age_range', 'Gender', 'Port', 'Has_cabin',
                   'Fare_range',  # 'SibSp',
                   'Prefix', 'Parch', 'Family_size', 'Pclass', ]

    train_y = train['Survived'].ravel()

    train_x = train[select_list].values
    test_set = test[select_list].values

    # 2.训练模型
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(train_x, train_y)

    print(clf.feature_importances_, '\n', train[select_list].columns)
    print(clf.score(train_x, train_y))

    # 3. 画图
    # 3.1 皮尔逊系数
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train[select_list].astype(float).corr(), linewidths=0.1, vmax=1.0,
                cmap=colormap, square=True, linecolor='white', annot=True)

    #
    # 3.2 性别和幸存
    sex_live_group = train.groupby(['Survived', 'Gender'])['PassengerId'].count().reset_index()
    sex_live_table = pd.pivot_table(sex_live_group, values='PassengerId', index='Survived', columns=['Gender'])
    sex_live_table.columns = ['Male', 'Female']

    sex_live_table.plot(kind='bar', colormap=colormap)
    plt.legend(loc='best', title='Gender')
    plt.xticks([0, 1], ['not lived', 'live'], rotation=0)

    #
    # 3.3 名称前缀和幸存
    prefix_live_group = train.groupby(['Survived', 'Prefix'])['PassengerId'].count().reset_index()
    prefix_live_table = pd.pivot_table(prefix_live_group, values='PassengerId', index='Survived', columns=['Prefix'])
    prefix_live_table.columns = ["Mr", "Miss", "Mrs", "Master", "Rare"]
    print(prefix_live_table)

    prefix_live_table.plot(kind='barh', colormap=colormap)
    plt.legend(loc='best', title='Prefix')
    plt.yticks([0, 1], ['not lived', 'live'])

    #
    # 3.4 和幸存

    plt.show()

