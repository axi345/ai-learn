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


# 粒子群算法
class PSO:
    def __init__(self, func=None, param_len=1, size=50, c1=2., c2=2., x_min=-10., x_max=10., v_min=-0.5, v_max=0.5,
                 r1=None, r2=None):
        self.func = func  # 计算适应性系数的方法
        self.param_len = param_len  # 参数个数
        self.size = size  # 有多少微粒
        self.c1 = c1  # 认知系数
        self.c2 = c2  # 社会系数
        self.history = []  # 每次迭代的最优值
        # 最小位移
        if type(x_min) != list:
            self.x_min = [x_min] * self.param_len
        else:
            assert len(x_min) == self.param_len  # 参数个数错误
            self.x_min = x_min
        # 最大位移
        if type(x_max) != list:
            self.x_max = [x_max] * self.param_len
        else:
            assert len(x_max) == self.param_len  # 参数个数错误
            self.x_max = x_max
        # 最小速度
        if type(v_min) != list:
            self.v_min = [v_min] * self.param_len
        else:
            assert len(v_min) == self.param_len  # 参数个数错误
            self.v_min = v_min
        # 最大速度
        if type(v_max) != list:
            self.v_max = [v_max] * self.param_len
        else:
            assert len(v_max) == self.param_len  # 参数个数错误
            self.v_max = v_max
        self.r1 = r1  # 随机数1
        self.r2 = r2  # 随机数2
        self.x = None  # 位移
        self.v = None  # 速度
        self.best_all_x = None  # 全局最优位置
        self.best_all_score = None  # 全局最优分数
        self.best_each_x = None  # 局部最优位置
        self.best_each_score = None  # 局部最优分数
        self._init_fit()

    def _init_fit(self):
        self.x = np.zeros([self.size, self.param_len])  # 形状为[微粒个数,参数个数]
        self.v = np.zeros([self.size, self.param_len])
        for i in range(self.size):
            for j in range(self.param_len):
                self.x[i][j] = np.random.uniform(self.x_min[j], self.x_max[j])
                self.v[i][j] = np.random.uniform(self.v_min[j], self.v_max[j])
        self.best_all_x = np.zeros(self.param_len)  # 全局最优位置
        self.best_all_score = -np.inf  # 全局最优分数
        self.best_each_x = self.x.copy()  # 局部最优位置
        self.best_each_score = np.full(self.size, -np.inf)  # 局部最优分数

    def solve(self, epoch=50, verbose=False):
        self.history = []
        r1 = self.r1
        r2 = self.r2
        for _ in range(epoch):  # 一共迭代_次
            # 配置随机变量
            if r1 is None:
                r1 = np.random.uniform(0, 1)
            if r2 is None:
                r2 = np.random.uniform(0, 1)
            # 计算适应度
            for i in range(self.size):  # 对于第i个微粒
                fitness = self.func(*self.x[i])
                # 更新局部最优值、局部最优位置
                if fitness > self.best_each_score[i]:
                    self.best_each_score[i] = fitness
                    self.best_each_x[i] = self.x[i].copy()
                if fitness > self.best_all_score:
                    self.best_all_score = fitness
                    self.best_all_x = self.x[i].copy()
            # 更新微粒的速度和位置
            self.v = self.v + self.c1 * r1 * (self.best_each_x - self.x) + self.c2 * r2 * (self.best_all_x - self.x)
            for j in range(self.param_len):
                self.v[:, j] = np.clip(self.v[:, j], self.v_min[j], self.v_max[j])
            self.x = self.x + self.v
            for j in range(self.param_len):
                self.x[:, j] = np.clip(self.x[:, j], self.x_min[j], self.x_max[j])
            if verbose:
                # print('已完成第%i次寻找,最优参数值为' % (_ + 1), self.best_all_x, '目前最优适合度为%.4f' % self.best_all_score)
                print('Number of iterations: %i.' % (_ + 1), 'Best fitness: %.4f' % self.best_all_score)
            self.history.append(self.best_all_score)
        self.history = np.array(self.history)
        return self.best_all_x
