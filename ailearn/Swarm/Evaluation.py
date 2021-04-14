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
    def __init__(self, a=20, b=0.2, c=2 * np.pi):
        self.min = -32.768
        self.max = 32.768
        self.a = a
        self.b = b
        self.c = c

    def func(self, *x):
        x = np.array(x)
        mean1 = np.mean(np.square(x))
        mean2 = np.mean(np.cos(self.c * x))
        result = -self.a * np.exp(-self.b * np.sqrt(mean1)) - np.exp(mean2) + self.a + np.e
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
        self.min = -5
        self.max = 10

    def func(self, x1, x2):
        result = np.square(1 - x1) + 100 * np.square(x2 - np.square(x1))
        return -result


# Schwefel函数
class Schwefel:
    def __init__(self):
        self.min = -500
        self.max = 500

    def func(self, *x):
        x = np.array(x)
        sum = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        result = 418.9829 * x.shape[0] - sum
        return -result


# Sphere函数
class Sphere:
    def __init__(self):
        self.min = -100
        self.max = 100

    def func(self, *x):
        x = np.array(x)
        result = np.sum(np.square(x))
        return -result


# Schwefel's Problem 1.2
class Schwefel_1:
    def __init__(self):
        self.min = -100
        self.max = 100

    def func(self, *x):
        x = np.array(x)
        a = np.zeros_like(x)
        for i in range(x.shape[0]):
            a[i] = x[:i].sum() ** 2
        return -a.sum()


# Schwefel's Problem 2.22
class Schwefel_2:
    def __init__(self):
        self.min = -10
        self.max = 10

    def func(self, *x):
        x = np.array(x)
        return -np.sum(np.abs(x)) - np.prod(np.abs(x))


# Branins函数
class Branins:
    def __init__(self):
        self.min = [-5, 0]
        self.max = [10, 15]

    def func(self, x1, x2):
        result = (x2 - (5.1 / (4 * np.pi ** 2)) + 5 / np.pi * x1 - 6) ** 2 + 10 * (1 - 1 / 8 / np.pi) * np.cos(x1) + 10
        return -result
