"""
    特征选择方法:
        1. 皮皮尔森相关系数，用于衡量变量之间的相关系系数，缺陷是无法较好的处理非线性的。

"""

import numpy as np
from scipy.stats import pearsonr

np.random.seed(0)
size = 300

x = np.random.normal(0, 1, size)
y = np.random.normal(0, 1, size)

# 返回相关系数和p-value
lower_noise = pearsonr(x, x + np.random.normal(0, 1, size))
high_noise = pearsonr(x, x + np.random.normal(0, 10, size))

print(lower_noise)
print(high_noise)
