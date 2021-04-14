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
# 若干进化算法的Python实现
import numpy as np


# 进化策略
class ES:
    def __init__(self, func=None, param_len=1, size=50, x_min=-10., x_max=10., alpha=0.5):
        self.func = func  # 计算适应性系数的方法
        self.param_len = param_len  # 参数个数
        self.size = size  # 有多少元素
        self.alpha = alpha  # 淘汰率
        self.history = []  # 每次迭代的最优值
        assert np.floor(self.size * (1 - self.alpha)) > 1
        # 最小数值
        if type(x_min) != list:
            self.x_min = [x_min] * self.param_len
        else:
            assert len(x_min) == self.param_len
            self.x_min = x_min
        # 最大数值
        if type(x_max) != list:
            self.x_max = [x_max] * self.param_len
        else:
            assert len(x_max) == self.param_len
            self.x_max = x_max
        self.x = None  # 数值
        self.v = None  # 变化幅度
        self.best_all_x = None  # 全局最优位置
        self.best_all_score = None  # 全局最优分数
        self.score = None  # 得分
        self._init_fit()

    def _init_fit(self):
        self.x = np.zeros([self.size, self.param_len])  # 形状为[微粒个数,参数个数]
        self.v = np.random.uniform(size=[self.size, self.param_len])
        self.score = np.zeros(self.size)
        for i in range(self.size):
            for j in range(self.param_len):
                self.x[i][j] = np.random.uniform(self.x_min[j], self.x_max[j])
        self.best_all_x = np.zeros(self.param_len)  # 全局最优位置
        self.best_all_score = -np.inf  # 全局最优分数

    def solve(self, epoch=50, verbose=False):
        self.history = []
        for _ in range(epoch):  # 一共迭代_次
            self.score = np.zeros(self.size)
            # 计算适应度
            for i in range(self.size):  # 对于第i个微粒
                self.score[i] = self.func(*self.x[i])
            # 更新局部最优值、局部最优位置
            if np.max(self.score) > self.best_all_score:
                self.best_all_score = np.max(self.score)
                self.best_all_x = self.x[np.argmax(self.score)]
            # 优胜劣汰
            for i in range(int(self.size * self.alpha)):
                idx = np.argmin(self.score)
                self.x = np.delete(self.x, idx, axis=0)
                self.v = np.delete(self.v, idx, axis=0)
                self.score = np.delete(self.score, idx, axis=0)
            # 繁殖
            x = np.zeros([self.size, self.param_len])
            v = np.zeros([self.size, self.param_len])
            for i in range(self.size):
                a, b = np.random.choice(self.x.shape[0], 2, False)
                mean = (self.x[a] + self.x[b]) / 2
                v[i] = np.abs(np.random.normal((self.v[a] + self.v[b]) / 2, 0.1))
                x[i] = np.clip(np.random.normal(mean, v[i]), self.x_min, self.x_max)
            self.x = x.copy()
            self.v = v.copy()
            if verbose:
                print('Number of iterations: %i.' % (_ + 1), 'Best fitness: %.4f' % self.best_all_score)
            self.history.append(self.best_all_score)
        self.history = np.array(self.history)
        return self.best_all_x
