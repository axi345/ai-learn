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
# 智能算法的若干不同的评价函数
import numpy as np


# Ackley函数
class Ackley:
    def __init__(self):
        self.min = -5
        self.max = 5

    def func(self, *x):
        x = np.array(x)
        mean1 = np.mean(np.square(x))
        mean2 = np.mean(np.cos(2 * np.pi * x))
        result = -20 * np.exp(-0.2 * np.sqrt(mean1)) - np.exp(mean2) + 20 + np.e
        return -result


# Griewank函数
class Griewank:
    def __init__(self):
        self.min = -600
        self.max = 600

    def func(self, *x):
        x = np.array(x)
        first = np.sum(np.square(x)) / 4000
        second = 1
        for i in range(x.shape[0]):
            second *= np.cos(x[i] / np.sqrt(i + 1))
        result = first - second + 1
        return -result


# Rastrigin函数
class Rastrigin:
    def __init__(self):
        self.min = -5.12
        self.max = 5.12

    def func(self, *x):
        x = np.array(x)
        sum = np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))
        result = 10 * x.shape[0] + sum
        return -result


# Rosenbrock函数
class Rosenbrock:
    def __init__(self):
        self.min = -100
        self.max = 100

    def func(self, x1, x2):
        result = np.square(1 - x1) + 100 * np.square(x2 - np.square(x1))
        return -result


# Schwefel函数
class Schwefel:
    def __init__(self):
        self.min = 0
        self.max = 500

    def func(self, *x):
        x = np.array(x)
        sum = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        result = 418.9829 * x.shape[0] - sum
        return -result
