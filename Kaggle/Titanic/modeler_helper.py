import numpy as np
from sklearn.model_selection import KFold

SEED = 0
NFOLDS = 5


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(self, x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


def get_oof(clf, x_train, y_train, x_test, ntrain, ntest):
    kf = KFold(NFOLDS, random_state=SEED)

    oof_train = np.zeros((ntrain))
    oof_test = np.zeros((ntest))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for train_index, test_index in enumerate(kf):
        pass

