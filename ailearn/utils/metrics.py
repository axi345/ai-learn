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
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# 准确率
def accuracy_score(prediction, label):
    '''
    :param prediction: 预测类别或one-hot编码
    :param label: 实际类别或one-hot编码
    :return: 准确率
    '''
    prediction, label = np.array(prediction), np.array(label)
    assert len(prediction.shape) == 1 or len(prediction.shape) == 2  # 输出值形状错误
    assert len(label.shape) == 1 or len(label.shape) == 2  # 真实值形状错误
    if len(prediction.shape) == 2:
        if prediction.shape[1] == 1:
            prediction = prediction.squeeze()
        else:
            prediction = np.argmax(prediction, 1)
    if len(label.shape) == 2:
        if label.shape[1] == 1:
            label = label.squeeze()
        else:
            label = np.argmax(label, 1)
    return np.mean(np.equal(prediction, label))


# 均方误差
def MSE(prediction, y, sample_importance=None):
    '''
    :param prediction: 预测值
    :param y: 真实值
    :param sample_importance: 样本重要性权重
    :return: 均方误差
    '''
    prediction, y = np.array(prediction).squeeze(), np.array(y).squeeze()
    assert len(prediction.shape) == 1 and len(y.shape) == 1  # 预测值或真实值形状错误
    if sample_importance is None:
        return np.mean(np.square(prediction - y))
    else:
        sample_importance = np.array(sample_importance)
        assert sample_importance.shape[0] == prediction.shape[0]  # 重要性权值形状错误
        return np.mean(sample_importance * np.square(prediction - y))


# 均方根误差
def RMSE(prediction, y, sample_importance=None):
    '''
    :param prediction: 预测值
    :param y: 真实值
    :param sample_importance: 样本重要性权重
    :return: 均方根误差
    '''
    prediction, y = np.array(prediction).squeeze(), np.array(y).squeeze()
    assert len(prediction.shape) == 1 and len(y.shape) == 1  # 预测值或真实值形状错误
    if sample_importance is None:
        return np.sqrt(np.mean(np.square(prediction - y)))
    else:
        sample_importance = np.array(sample_importance)
        assert sample_importance.shape[0] == prediction.shape[0]  # 重要性权值形状错误
        return np.sqrt(np.mean(sample_importance * np.square(prediction - y)))


# 平均绝对误差
def MAE(prediction, y, sample_importance=None):
    '''
    :param prediction: 预测值
    :param y: 真实值
    :param sample_importance: 样本重要性权重
    :return:
    '''
    prediction, y = np.array(prediction).squeeze(), np.array(y).squeeze()
    assert len(prediction.shape) == 1 and len(y.shape) == 1  # 预测值或真实值形状错误
    if sample_importance is None:
        return np.mean(np.abs(prediction - y))
    else:
        sample_importance = np.array(sample_importance)
        assert sample_importance.shape[0] == prediction.shape[0]  # 重要性权值形状错误
        return np.mean(sample_importance * np.abs(prediction - y))


# 误差平方和
def SSE(prediction, y, sample_importance=None):
    '''
    :param prediction: 预测值
    :param y: 真实值
    :param sample_importance: 样本重要性权重
    :return:
    '''
    prediction, y = np.array(prediction).squeeze(), np.array(y).squeeze()
    assert len(prediction.shape) == 1 and len(y.shape) == 1  # 预测值或真实值形状错误
    if sample_importance is None:
        return np.sum(np.square(prediction - y))
    else:
        sample_importance = np.array(sample_importance)
        assert sample_importance.shape[0] == prediction.shape[0]  # 重要性权值形状错误
        return np.sum(sample_importance * np.square(prediction - y))


# 总平方和
def SST(y, sample_importance=None):
    '''
    :param y: 真实值
    :param sample_importance: 样本重要性权重
    :return: 总平方和
    '''
    y = np.array(y)
    assert len(y.shape) == 1  # 真实值形状错误
    if sample_importance is None:
        return np.sum(np.square(y - np.mean(y)))
    else:
        sample_importance = np.array(sample_importance)
        assert sample_importance.shape[0] == y.shape[0]  # 重要性权值形状错误
        return np.sum(sample_importance * np.square(y - np.mean(y)))


# 回归平方和
def SSR(prediction, y, sample_importance=None):
    '''
    :param prediction: 预测值
    :param y: 真实值
    :param sample_importance: 样本重要性权重
    :return: 回归平方和
    '''
    prediction, y = np.array(prediction).squeeze(), np.array(y).squeeze()
    assert len(prediction.shape) == 1 and len(y.shape) == 1  # 预测值或真实值形状错误
    if sample_importance is None:
        return np.sum(np.square(prediction - np.mean(y)))  # Total sum of squares
    else:
        sample_importance = np.array(sample_importance)
        assert sample_importance.shape[0] == prediction.shape[0]  # 重要性权值形状错误
        return np.sum(sample_importance * np.square(prediction - np.mean(y)))


# 确定系数
def R_square(prediction, y, sample_importance=None):
    '''
    :param prediction: 预测值
    :param y: 真实值
    :param sample_importance: 样本重要性权重
    :return: 确定系数
    '''
    return 1 - SSE(prediction, y, sample_importance) / SST(y, sample_importance)


# 皮尔森相关系数
def PC(prediction, y):
    '''
    :param prediction: 预测值
    :param y: 真实值
    :return:
    '''
    prediction, y = np.array(prediction).squeeze(), np.array(y).squeeze()
    assert len(prediction.shape) == 1 and len(y.shape) == 1  # 预测值或真实值形状错误
    n = y.shape[0]
    return (n * np.sum(prediction * y) - np.sum(prediction) * np.sum(y)) / (
            np.sqrt(n * np.sum(prediction ** 2) - np.sum(prediction) ** 2) * np.sqrt(
        n * np.sum(y ** 2) - np.sum(y) ** 2))


# K折交叉验证
def cross_val_score(estimator, x, y, k=10, verbose=True, random_state=None, **kwargs):
    '''
    :param estimator: 待评价的模型
    :param x: 样本数据
    :param y: 样本标签
    :param k: K折交叉验证中的K值
    :param verbose: 是否显示验证过程
    :param random_state: 数据集分割的随机数种子
    :param kwargs: estimator.fit()的参数
    :return: k次验证的准确率一维数组
    '''
    x, y = np.array(x), np.array(y)
    if random_state is None:
        folder = StratifiedKFold(k, True)
    else:
        folder = StratifiedKFold(k, True, random_state)
    scores = []
    for i, (train_index, test_index) in enumerate(folder.split(x, y)):
        estimator.fit(x[train_index], y[train_index], **kwargs)
        score = estimator.score(x[test_index], y[test_index])
        scores.append(score)
        if verbose:
            print('第%d次交叉验证完成，得分为%.4f' % (i + 1, score))
    scores = np.array(scores)
    return scores


# 留p法交叉验证
def leave_p_score(estimator, x, y, p=1, verbose=True, **kwargs):
    '''
    :param estimator: 待评价的模型
    :param x: 样本数据
    :param y: 样本标签
    :param p: 留p法交叉验证中的p值
    :param verbose: 是否显示验证过程
    :param kwargs: estimator.fit()的参数
    :return: len(x)//p次验证的准确率一维数组
    '''
    x, y = np.array(x), np.array(y)
    if x.shape[0] < p:
        warnings.warn('交叉验证参数错误，不执行操作！')
        return None
    epoch = x.shape[0] // p
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    scores = []
    for i in range(epoch):
        test_index = slice(i * p, (i + 1) * p)
        train_index = np.delete(index, test_index)
        estimator.fit(x[train_index], y[train_index], **kwargs)
        score = estimator.score(x[test_index], y[test_index])
        scores.append(score)
        if verbose:
            print('第%d次交叉验证完成，得分为%.4f' % (i + 1, score))
    scores = np.array(scores)
    return scores


# Hold-Out检验
def hold_out_score(estimator, x, y, test_size=0.25, shuffle=True, stratify=None, random_state=0, **kwargs):
    '''
    :param estimator: 待评价的模型
    :param x: 样本数据
    :param y: 样本标签
    :param test_size: 测试数据比率
    :param shuffle: 是否随机化
    :param stratify: 是否分层采样(一般输入y)
    :param random_state: 数据集分割的随机数种子
    :param kwargs: estimator.fit()的参数
    :return: 准确率
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle, stratify=stratify,
                                                        random_state=random_state)
    estimator.fit(x_train, y_train, **kwargs)
    return estimator.score(x_test, y_test)


# 分类报告
def class_report(y_true, y_pred, method='weighted'):
    '''
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param method: 报告方式，包括'weighted'、'micro'、'macro'
    :return: 分类报告字典
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracy = np.mean(y_true == y_pred)
    cm = confusion_matrix(y_true, y_pred)  # 混淆矩阵
    _, _, _, num = precision_recall_fscore_support(y_true, y_pred)  # num为每一类的样本数目列表
    n_classes = len(num)
    n_samples = len(y_true)
    TP, FN, FP, TN = np.zeros([n_classes], int), np.zeros([n_classes], int), np.zeros([n_classes], int), np.zeros(
        [n_classes], int)  # 每一类的TP、FN、FP、TN
    precision, recall, recall_, F1_score, G_mean = np.zeros([n_classes]), np.zeros([n_classes]), np.zeros(
        [n_classes]), np.zeros([n_classes]), np.zeros([n_classes])  # 每一类的精确度、召回率、负类召回率、F1值、G-mean值
    # 微平均
    if method == 'micro':
        for i in range(n_classes):
            TP[i] = cm[i][i]
            FP[i] = np.sum(cm[:, i]) - TP[i]
            FN[i] = np.sum(cm[i]) - TP[i]
            TN[i] = n_samples - TP[i] - FP[i] - FN[i]
        TP, FP, FN, TN = np.mean(TP), np.mean(FP), np.mean(FN), np.mean(TN)
        if TP == 0 and FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if TP == 0 and FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        if TN == 0 and FP == 0:
            recall_ = 0
        else:
            recall_ = TN / (TN + FP)
        if recall == 0 and precision == 0:
            F1_score = 0
        else:
            F1_score = 2 * precision * recall / (precision + recall)
        if recall == 0 and recall_ == 0:
            G_mean = 0
        else:
            G_mean = np.sqrt(recall * recall_)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1_score': F1_score, 'G_mean': G_mean}

    # 宏平均或加权平均
    elif method == 'macro' or method == 'weighted':
        for i in range(n_classes):
            TP[i] = cm[i][i]
            FP[i] = np.sum(cm[:, i]) - TP[i]
            FN[i] = np.sum(cm[i]) - TP[i]
            TN[i] = n_samples - TP[i] - FP[i] - FN[i]
            if TP[i] == 0 and FP[i] == 0:
                precision[i] = 0
            else:
                precision[i] = TP[i] / (TP[i] + FP[i])
            if TP[i] == 0 and FN[i] == 0:
                recall[i] = 0
            else:
                recall[i] = TP[i] / (TP[i] + FN[i])
            if TN[i] == 0 and FP[i] == 0:
                recall_[i] = 0
            else:
                recall_[i] = TN[i] / (TN[i] + FP[i])
            if recall[i] == 0 and precision[i] == 0:
                F1_score[i] = 0
            else:
                F1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            if recall[i] == 0 and recall_[i] == 0:
                G_mean[i] = 0
            else:
                G_mean[i] = np.sqrt(recall[i] * recall_[i])
        if method == 'macro':
            precision = np.mean(precision)
            recall = np.mean(recall)
            F1_score = np.mean(F1_score)
            G_mean = np.mean(G_mean)
            return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1_score': F1_score,
                    'G_mean': G_mean}
        else:
            precision = np.sum(precision * num) / n_samples
            recall = np.sum(recall * num) / n_samples
            F1_score = np.sum(F1_score * num) / n_samples
            G_mean = np.sum(G_mean * num) / n_samples
            return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1_score': F1_score,
                    'G_mean': G_mean}

    else:
        print('Classification report type error!')
