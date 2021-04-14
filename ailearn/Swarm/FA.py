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


# 萤火虫算法
class FA:
    def __init__(self, func=None, param_len=1, size=50, x_min=-10., x_max=10., beta=2, alpha=0.5, gamma=0.9):
        self.func = func  # 计算适应性系数的方法
        self.param_len = param_len  # 参数个数
        self.size = size  # 有多少萤火虫
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
        self.alpha = alpha  # 步长
        self.beta = beta  # 最大吸引度
        self.gamma = gamma  # 光强吸收系数
        self.x = None  # 位移
        self.score = None  # 每个萤火虫的分数
        self.best_all_x = None  # 全局最优位置
        self.best_all_score = None  # 全局最优分数
        self._init_fit()

    def _init_fit(self):
        self.x = np.zeros([self.size, self.param_len])  # 形状为[萤火虫的个数,参数个数]
        self.score = np.zeros(self.size)
        for i in range(self.size):
            for j in range(self.param_len):
                self.x[i][j] = np.random.uniform(self.x_min[j], self.x_max[j])
            self.score[i] = self.func(*self.x[i])
        self.best_all_score = np.max(self.score)  # 全局最优分数
        self.best_all_x = self.x[np.argmax(self.score)]  # 全局最优位置

    def solve(self, epoch=50, verbose=False):
        self.history = []
        for _ in range(epoch):  # 一共迭代_次
            # 计算距离
            distance = np.zeros([self.size, self.size])
            for i in range(self.size):
                for j in range(self.size - 1, i, -1):
                    distance[i][j] = np.linalg.norm(self.x[i] - self.x[j], 2)
                    distance[j][i] = distance[i][j]
            for i in range(self.size):  # 对于第i个萤火虫
                # 对于最亮的萤火虫，做随机运动
                if np.argmax(self.score) == i:
                    self.x[i] += self.alpha * np.random.uniform(-0.5, 0.5)
                    self.x[i] = np.clip(self.x[i], self.x_min, self.x_max)
                    self.score[i] = self.func(*self.x[i])
                    # 更新局部最优值、局部最优位置
                    if self.score[i] > self.best_all_score:
                        self.best_all_score = self.score[i]
                        self.best_all_x = self.x[i].copy()
                    continue
                lightness = np.zeros(self.size)
                # 查找对自己相对亮度最大的萤火虫
                for j in range(self.size):
                    if j == i:
                        lightness[j] = -np.inf
                    else:
                        lightness[j] = self.score[j] * np.exp(-self.gamma * distance[i][j])
                idx = np.argmax(lightness)
                # 更新萤火虫的位置
                self.x[i] += self.beta * np.exp(-self.gamma * np.square(distance[i][idx])) * (
                        self.x[idx] - self.x[i]) + self.alpha * np.random.uniform(-0.5, 0.5)
                self.score[i] = self.func(*self.x[i])
                self.x[i] = np.clip(self.x[i], self.x_min, self.x_max)
                # 更新局部最优值、局部最优位置
                if self.score[i] > self.best_all_score:
                    self.best_all_score = self.score[i]
                    self.best_all_x = self.x[i].copy()
            if verbose:
                # print('已完成第%i次寻找,最优参数值为' % (_ + 1), self.best_all_x, '目前最优适合度为%.4f' % self.best_all_score)
                print('Number of iterations: %i. The optimal parameters:' % (_ + 1), self.best_all_x,
                      'Best fitness: %.4f' % self.best_all_score)
            self.history.append(self.best_all_score)
        self.history = np.array(self.history)
        return self.best_all_x
