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

# -*- coding: utf-8 -*-
import numpy as np
from ..utils import to_categorical
from .activations import softmax, sigmoid


# softmax交叉熵
def softmax_cross_entropy(out, label):
    # out:神经元的输出值
    # label:实际类别或one-hot编码
    out, label = np.array(out), np.array(label)
    assert len(out.shape) == 2  # 输出形状错误
    assert len(label.shape) == 1 or len(label.shape) == 2  # 标签形状错误
    if len(label.shape) == 1:  # 转化为one-hot编码
        y = to_categorical(label, num_classes=out.shape[1])
    else:
        if label.shape[1] == 1:
            y = to_categorical(label.squeeze(), num_classes=out.shape[1])
        else:
            assert label.max() == 1 and label.sum(1).mean() == 1  # 标签one-hot编码错误
            y = label
    yhat = softmax(out)
    return -np.mean(y * np.log(yhat))


# 交叉熵
def cross_entropy(out, label):
    # out:神经元的输出值
    # label:实际类别或one-hot编码
    yhat, label = np.array(out), np.array(label)
    assert len(out.shape) == 2  # 输出形状错误
    assert len(label.shape) == 1 or len(label.shape) == 2  # 标签形状错误
    if len(label.shape) == 1:  # 转化为one-hot编码
        y = to_categorical(label, num_classes=out.shape[1])
    else:
        if label.shape[1] == 1:
            y = to_categorical(label.squeeze(), num_classes=out.shape[1])
        else:
            assert label.max() == 1 and label.sum(1).mean() == 1  # 标签one-hot编码错误
            y = label
    return -np.mean(y * np.log(yhat))


# 二分类
def sigmoid_binary_cross_entropy(out, label):
    # out:神经元的输出值
    # label:实际类别或one-hot编码
    out, y = np.array(out), np.array(label)
    assert len(out.shape) == 2 and out.shape[1] == 1  # 输出形状错误
    assert len(y.shape) == 1  # 标签形状错误
    yhat = sigmoid(out)
    return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))


# 二分类
def binary_cross_entropy(out, label):
    # out:神经元的输出值
    # label:实际类别或one-hot编码
    yhat, y = np.array(out), np.array(label)
    assert len(yhat.shape) == 2 and out.shape[1] == 1  # 输出形状错误
    assert len(y.shape) == 1  # 标签形状错误
    return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))


# 最小二乘损失
def square_loss(prediction, y):
    # prediction:预测值
    # y:实际值
    prediction, y = np.array(prediction), np.array(y)
    assert (len(prediction.shape) == 2 and prediction.shape[1] == 1) or len(prediction.shape) == 1  # 输出形状错误
    assert len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)  # 真实值形状错误
    return np.sum(np.sum(np.square(prediction.reshape(-1, 1) - y.reshape(-1, 1)), 1))


# 均方误差
def mse(prediction, y):
    # prediction:预测值
    # y:实际值
    prediction, y = np.array(prediction), np.array(y)
    assert (len(prediction.shape) == 2 and prediction.shape[1] == 1) or len(prediction.shape) == 1  # 输出形状错误
    assert len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)  # 真实值形状错误
    return np.mean(np.sum(np.square(prediction.reshape(-1, 1) - y.reshape(-1, 1)), 1))
