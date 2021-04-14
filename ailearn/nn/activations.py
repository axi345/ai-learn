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


# relu激活函数
def relu(x):
    x = np.array(x)
    return np.maximum(0, x)


# tanh激活函数
def tanh(x):
    x = np.array(x)
    return np.tanh(x)


# sigmoid激活函数
def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))


# softmax激活函数
def softmax(x):
    x = np.array(x)
    assert len(x.shape) == 1 or len(x.shape) == 2
    if len(x.shape) == 1:
        x = x - x.max()
        x = np.exp(x)
        return x / x.sum()
    else:
        x = x - x.max(1, keepdims=True)
        x = np.exp(x)
        return x / x.sum(1, keepdims=True)


# linear激活函数
def linear(x):
    x = np.array(x)
    return x


# 阈值激活函数
def threshold(x, threshold=0):
    x = np.array(x)
    out = np.zeros_like(x, dtype=np.float)
    out[x >= threshold] = 1
    return out


# arctan激活函数
def arctan(x):
    x = np.array(x)
    return np.arctan(x)


# leaky relu
def leaky_relu(x, alpha=0.1):
    x = np.array(x, dtype=np.float)
    x[x < 0] = (x * alpha)[x < 0]
    return x


# prelu激活函数
def prelu(x, p):
    x = np.array(x, dtype=np.float)
    x[x < 0] = (x * p)[x < 0]
    return x


# elu激活函数
def elu(x, alpha=0.1):
    x = np.array(x, dtype=np.float)
    x[x < 0] = (alpha * (np.exp(x) - 1))[x < 0]
    return x


# softplus激活函数
def softplus(x):
    x = np.array(x)
    return np.log(1 + np.exp(x))


# bent identity
def bent_identity(x):
    x = np.array(x)
    return (np.sqrt(np.square(x) + 1) - 1) * 0.5 + x


# Soft Exponential
def soft_exponential(x, p):
    x = np.array(x, dtype=np.float)
    x[p < 0] = (-np.log(np.maximum(1 - p[p < 0] * (x[p < 0] + p[p < 0]), 1e-7)) / p[p < 0])
    x[p == 0] = 0
    x[p > 0] = ((np.exp(p * x) - 1) / p + p)[p > 0]
    return x


# Sinusoid
def sin(x):
    x = np.array(x)
    return np.sin(x)


# Sinc
def sinc(x):
    x = np.array(x, dtype=np.float)
    out = np.ones_like(x, dtype=np.float)
    out[x != 0] = np.sin(x[x != 0]) / x[x != 0]
    return out


# Gaussian
def guassian(x):
    x = np.array(x)
    return np.exp(-np.square(x))
