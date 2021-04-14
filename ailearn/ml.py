# -*- coding: utf-8 -*-
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
import warnings
from .utils import to_categorical
from .nn.activations import *
from sklearn.metrics import r2_score


# 核方法（A:[m1,n],B:[m2,n],K:[m1,m2]）
def Kernel(A, B=None, kernel='linear', gamma=0.1, q=0.1, degree=3):
    '''
    :param A: 矩阵1 ，形状为[m1,n]
    :param B: 矩阵2，形状为[m2,n]，取值为None时B=A.copy()
    :param kernel: 核的类型，['linear','polynomial'('poly'),'sigmoid','rbf','laplace']
    :param gamma: 多项式核、Sigmoid核、高斯核、拉普拉斯核中的参数
    :param q: 多项式核、Sigmoid核、高斯核中的参数
    :param degree: 多项式核中的参数
    :return: 核运算得到的矩阵[m1,m2]
    '''
    A = np.array(A, np.float)
    if B is None:
        B = A.copy()
    B = np.array(B, np.float)
    if kernel == 'linear':  # 线性核
        my_kernel = A.dot(B.T)
    elif kernel == 'polynomial' or kernel == 'poly':  # 多项式核
        my_kernel = (gamma * A.dot(B.T) + q) ** degree
    elif kernel == 'sigmoid':  # Sigmoid核
        my_kernel = np.tanh(gamma * A.dot(B.T) - q)
    elif kernel == 'rbf':  # 高斯核
        rA = np.sum(np.square(A), 1, keepdims=True)
        rB = np.sum(np.square(B), 1, keepdims=True)
        sq_dists = rA - 2 * A.dot(B.T) + np.transpose(rB)  # x^2-2*x*y+y^2
        my_kernel = np.exp(-gamma * np.abs(sq_dists))
    elif kernel == 'laplace':  # 拉普拉斯核
        A, B = np.array(A), np.array(B)
        rA, rB = np.expand_dims(A, 1), np.expand_dims(B, 0)
        my_kernel = np.exp(-gamma * np.sum(np.abs(rA - rB), axis=2))
    else:
        print('kernel error!')
        return
    return my_kernel


# 主成分分析
def PCA(x, feature_num=None, svd=False):
    '''
    :param x: 样本矩阵，形状为[样本个数,特征维数]
    :param feature_num: 保留的特征数，为None时等于特征维数
    :param svd: 是否使用svd进行PCA
    :return: 计算后得到的矩阵，形状为[样本个数,feature_num]
    '''
    x = np.array(x, np.float)
    if len(x.shape) != 1 and len(x.shape) != 2:
        warnings.warn('数据维度不正确，不执行操作！')
        return None
    if len(x.shape) == 1:
        x = np.expand_dims(x, 0)
    if feature_num is None:
        feature_num = x.shape[1]
    x -= x.mean(0, keepdims=True)
    if svd:
        U, S, VT = np.linalg.svd(x)
        index_sort = np.argsort(S)  # 对奇异值进行排序
        index = index_sort[-1:-(feature_num + 1):-1]
        return x.dot(VT[index])  # 乘上最大的feature_num个奇异值组成的特征向量
    else:
        eigval, eigvec = np.linalg.eig(x.transpose().dot(x) / x.shape[0])
        index_sort = np.argsort(eigval)  # 对特征值进行排序
        index = index_sort[-1:-(feature_num + 1):-1]
        return x.dot(eigvec[:, index])  # 乘上最大的feature_num个特征组成的特征向量


# 核化主成分分析
def KernelPCA(x, feature_num=None, kernel='rbf', gamma=0.1, q=0.1, degree=3):
    # x：样本
    # feature_num：保留的特征数
    x = np.array(x, np.float)
    if len(x.shape) != 1 and len(x.shape) != 2:
        warnings.warn('数据维度不正确，不执行操作！')
        return None
    if len(x.shape) == 1:
        x = np.expand_dims(x, 0)
    if feature_num is None:
        feature_num = x.shape[1]
    K = Kernel(x, kernel=kernel, gamma=gamma, q=q, degree=degree)
    one_m = np.ones([x.shape[0], x.shape[0]]) / x.shape[0]
    K = K - one_m.dot(K) - K.dot(one_m) + one_m.dot(K).dot(one_m)
    eigval, eigvec = np.linalg.eig(K)
    index_sort = np.argsort(eigval)  # 对特征值进行排序
    index = index_sort[-1:-(feature_num + 1):-1]
    lambdas = eigval[index]
    alphas = eigvec[:, index]  # 乘上最大的feature_num个特征组成的特征向量
    return K.dot(alphas / np.sqrt(lambdas))


# ELM分类器
class ELMClassifier:
    def __init__(self, n_hidden=150, activation='leaky_relu', kernel=None, gamma=0.1, q=0.1, degree=3):
        self.n_hidden = n_hidden  # 不使用核时，隐层节点个数
        self.activation = activation.lower()  # 不使用核时，激活函数
        self.w1, self.b, self.w2 = None, None, None  # 参数值
        self.kernel = kernel  # 核的类型，[None,'linear','polynomial'('poly'),'sigmoid','rbf','laplace']
        self.gamma = gamma  # 多项式核、Sigmoid核、高斯核、拉普拉斯核中的参数
        self.q = q  # 多项式核、Sigmoid核、高斯核中的参数
        self.degree = degree  # 多项式核中的参数
        self.x_train = None  # 训练数据

    def fit(self, x_train, y_train):
        x_train, y_train = np.array(x_train), np.array(y_train)
        self.x_train = x_train.copy()
        if self.kernel == None:
            self.w1 = np.random.uniform(-1, 1, (x_train.shape[1], self.n_hidden))
            self.b = np.random.uniform(-1, 1, self.n_hidden)
            H = eval('%s(x_train.dot(self.w1) + self.b)' % self.activation)
            self.w2 = np.linalg.pinv(H).dot(to_categorical(y_train))
        else:
            H = Kernel(x_train, self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q, degree=self.degree)
            self.w2 = np.linalg.pinv(H).dot(to_categorical(y_train))

    def predict(self, x_test):
        x_test = np.array(x_test)
        if self.kernel == None:
            x_test = eval(' %s(x_test.dot(self.w1) + self.b).dot(self.w2)' % self.activation)
        else:
            x_test = Kernel(x_test, self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q,
                            degree=self.degree).dot(self.w2)
        return np.argmax(x_test, axis=1)

    def score(self, x_test, y_test):
        x_test, y_test = np.array(x_test), np.array(y_test)
        y_pred = self.predict(x_test)
        return np.mean(np.equal(y_test, y_pred))

    # 保存模型
    def save(self, filepath):
        np.savez(filepath, n_hidden=self.n_hidden, activation=self.activation, w1=self.w1, b=self.b, w2=self.w2,
                 kernel=self.kernel, gamma=self.gamma, q=self.q, degree=self.degree, x_train=self.x_train)

    # 读取模型
    def load(self, filepath):
        if '.npz' not in filepath:
            filepath = filepath + '.npz'
        params = np.load(filepath)
        self.n_hidden = params['n_hidden']
        self.activation = params['activation']
        self.w1, self.b, self.w2 = params['w1'], params['b'], params['w2']
        self.kernel = params['kernel']
        self.gamma, self.q, self.degree = params['gamma'], params['q'], params['degree']
        self.x_train = params['x_train']


# ELM回归器
class ELMRegressor:
    def __init__(self, n_hidden=150, activation='leaky_relu', kernel=None, gamma=0.1, q=0.1, degree=3):
        self.n_hidden = n_hidden  # 不使用核时，隐层节点个数
        self.activation = activation.lower()  # 不使用核时，激活函数
        self.w1, self.b, self.w2 = None, None, None  # 参数值
        self.kernel = kernel  # 核的类型，[None,'linear','polynomial'('poly'),'sigmoid','rbf','laplace']
        self.gamma = gamma  # 多项式核、Sigmoid核、高斯核、拉普拉斯核中的参数
        self.q = q  # 多项式核、Sigmoid核、高斯核中的参数
        self.degree = degree  # 多项式核中的参数
        self.x_train = None  # 训练数据

    def fit(self, x_train, y_train):
        x_train, y_train = np.array(x_train), np.array(y_train).reshape(-1, 1)
        self.x_train = x_train.copy()
        if self.kernel == None:
            self.w1 = np.random.uniform(-1, 1, (x_train.shape[1], self.n_hidden))
            self.b = np.random.uniform(-1, 1, self.n_hidden)
            H = eval('%s(x_train.dot(self.w1) + self.b)' % self.activation)
            self.w2 = np.linalg.pinv(H).dot(y_train)
        else:
            H = Kernel(x_train, self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q, degree=self.degree)
            self.w2 = np.linalg.pinv(H).dot(y_train)

    def predict(self, x_test):
        x_test = np.array(x_test)
        if self.kernel == None:
            return eval(' %s(x_test.dot(self.w1) + self.b).dot(self.w2)' % self.activation)
        else:
            return Kernel(x_test, self.x_train, kernel=self.kernel, gamma=self.gamma, q=self.q,
                          degree=self.degree).dot(self.w2)

    def score(self, x_test, y_test):
        x_test, y_test = np.array(x_test), np.array(y_test)
        y_pred = self.predict(x_test)
        return r2_score(y_test, y_pred)

    # 保存模型
    def save(self, filepath):
        np.savez(filepath, n_hidden=self.n_hidden, activation=self.activation, w1=self.w1, b=self.b, w2=self.w2,
                 kernel=self.kernel, gamma=self.gamma, q=self.q, degree=self.degree, x_train=self.x_train)

    # 读取模型
    def load(self, filepath):
        if '.npz' not in filepath:
            filepath = filepath + '.npz'
        params = np.load(filepath)
        self.n_hidden = params['n_hidden']
        self.activation = params['activation']
        self.w1, self.b, self.w2 = params['w1'], params['b'], params['w2']
        self.kernel = params['kernel']
        self.gamma, self.q, self.degree = params['gamma'], params['q'], params['degree']
        self.x_train = params['x_train']


