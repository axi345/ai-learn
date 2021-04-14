# Copyright 2018 Zhao Xingyu & An Yuexuan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
import numpy as np
import warnings


# 闵可夫斯基距离
def minkowski_distance(a, b, p):
    '''
    :param a: 向量1
    :param b: 向量2
    :param p: 参数（阶数）
    :return: 闵可夫斯基距离
    '''
    a = np.array(a).squeeze()
    b = np.array(b).squeeze()
    if len(a.shape) != 1 or len(b.shape) != 1:
        warnings.warn('数据维度不为1，不执行操作！')
        return None
    return np.power(np.sum(np.power(np.abs(a - b), p)), 1 / p)


# l1范数
def l1_distance(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: l1范数
    '''
    return minkowski_distance(a, b, 1)


# 曼哈顿距离
def manhattan_distance(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: 曼哈顿距离
    '''
    return minkowski_distance(a, b, 1)


# l2范数
def l2_distance(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: l2范数
    '''
    return minkowski_distance(a, b, 2)


# 欧拉距离
def euclidean_distance(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: 欧拉距离
    '''
    return minkowski_distance(a, b, 2)


# 切比雪夫距离
def chebyshev_distance(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: 切比雪夫距离
    '''
    a = np.array(a).squeeze()
    b = np.array(b).squeeze()
    if len(a.shape) != 1 or len(b.shape) != 1:
        warnings.warn('数据维度不为1，不执行操作！')
        return None
    return np.max(np.abs(a - b))


# 夹角余弦
def cosine(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: 夹角余弦
    '''
    a = np.array(a).squeeze()
    b = np.array(b).squeeze()
    if len(a.shape) != 1 or len(b.shape) != 1:
        warnings.warn('数据维度不为1，不执行操作！')
        return None
    return a.dot(b) / (np.linalg.norm(a, 2) * np.linalg.norm(b, 2))


# 汉明距离
def hamming_distance(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: 汉明距离
    '''
    a = np.array(a, np.str).squeeze()
    b = np.array(b, np.str).squeeze()
    if len(a.shape) != 1 or len(b.shape) != 1:
        warnings.warn('数据维度不为1，不执行操作！')
        return None
    return np.sum(a != b)


# 杰拉德相似系数
def jaccard_similarity_coefficient(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: 杰拉德相似系数
    '''
    a = set(a)
    b = set(b)
    return len(a.intersection(b)) / len(a.union(b))


# 杰拉德距离
def jaccard_distance(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: 杰拉德距离
    '''
    return 1 - jaccard_similarity_coefficient(a, b)


# 相关系数
def correlation_coefficient(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: 相关系数
    '''
    a = np.array(a).squeeze()
    b = np.array(b).squeeze()
    if len(a.shape) != 1 or len(b.shape) != 1:
        warnings.warn('数据维度不为1，不执行操作！')
        return None
    return ((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std())


# 相关距离
def correlation_distance(a, b):
    '''
    :param a: 向量1
    :param b: 向量2
    :return: 相关距离
    '''
    return 1 - correlation_coefficient(a, b)


# 马氏距离
def mahalanobis_distance(x):
    '''
    :param x: 样本矩阵，形状：[样本个数,特征维数]
    :return: 不同样本间的马氏距离
    '''
    x = np.array(x)
    if len(x.shape) != 2:
        warnings.warn('数据维度不为2，不执行操作！')
        return None
    m, n = x.shape
    S_inv = np.linalg.pinv(np.cov(x))
    D = np.zeros([m, m])
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = np.sqrt((x[i] - x[j]).reshape(1, n).dot(S_inv).dot((x[i] - x[j]).reshape(n, 1)))
            D[j, i] = D[i, j]
    return D


# 计算不同样本之间的距离，输入：[m,n]，输出[m,m]，其中dist[i,j]为x[i]到x[j]的距离
def distance_matrix(x):
    '''
    :param x: 样本矩阵，形状：[样本个数,特征维数]
    :return: 不同样本之间的距离
    '''
    m = x.shape[0]
    distance = np.zeros([m, m])
    for i in range(m):
        for j in range(m - 1, i, -1):
            distance[i][j] = np.linalg.norm(x[i] - x[j], 2)
            distance[j][i] = distance[i][j]
    return distance
