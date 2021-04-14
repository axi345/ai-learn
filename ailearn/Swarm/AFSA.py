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


# 人工鱼群算法
class AFSA:
    def __init__(self, func=None, param_len=1, size=50, x_min=-10., x_max=10., visual=1., step=0.5, delta=1,
                 try_number=5):
        self.func = func  # 计算适应性系数的方法
        self.param_len = param_len  # 参数个数
        self.size = size  # 有多少鱼
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
        self.visual = visual  # 鱼的视野
        self.step = step  # 鱼的步长
        self.delta = delta  # 拥挤因子
        self.try_number = try_number  # 觅食行为的尝试次数
        self.x = None  # 位移
        self.score = None  # 每条鱼的分数
        self.best_all_x = None  # 全局最优位置
        self.best_all_score = None  # 全局最优分数
        self._init_fit()

    def _init_fit(self):
        self.x = np.zeros([self.size, self.param_len])  # 形状为[鱼的个数,参数个数]
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
            for i in range(self.size):  # 对于第i条鱼
                # 检测周围有没有鱼
                fishes = []
                fish_idx = []
                fish_num = 0
                for j in range(self.size):  # 搜索附近有多少鱼
                    if np.sum(np.square(self.x[i] - self.x[j])) < self.visual:
                        fish_num += 1
                        fishes.append(self.x[j])
                        fish_idx.append(j)
                if fish_num > 1:  # 如果周围有鱼
                    fishes = np.array(fishes)
                    # 聚群行为
                    centre = np.mean(fishes, 0)
                    if self.func(*centre) / fish_num > self.delta * self.score[i]:
                        self.x[i] += (centre - self.x[i]) / (np.sum(
                            np.square(centre - self.x[i])) + 10e-8) * self.step * np.random.rand()
                        self.x[i] = np.clip(self.x[i], self.x_min, self.x_max)
                        self.score[i] = self.func(*self.x[i])
                        # 更新局部最优值、局部最优位置
                        if self.score[i] > self.best_all_score:
                            self.best_all_score = self.score[i]
                            self.best_all_x = self.x[i].copy()
                        continue
                    # 追尾行为
                    best_index = fish_idx[np.argmax(self.score[fish_idx])]
                    if self.score[best_index] / fish_num > self.delta * self.score[i]:
                        self.x[i] += (self.x[best_index] - self.x[i]) / np.sum(
                            np.square(self.x[best_index] - self.x[i])) * self.step * np.random.rand()
                        self.x[i] = np.clip(self.x[i], self.x_min, self.x_max)
                        self.score[i] = self.func(*self.x[i])
                        # 更新局部最优值、局部最优位置
                        if self.score[i] > self.best_all_score:
                            self.best_all_score = self.score[i]
                            self.best_all_x = self.x[i].copy()
                        continue
                # 觅食行为
                find = False
                x = None
                for j in range(self.try_number):
                    x = self.x[i] + self.visual * np.random.uniform(-1, 1, self.param_len)
                    x = np.clip(x, self.x_min, self.x_max)
                    if self.func(*x) > self.score[i]:
                        find = True
                    break
                if find is True:
                    self.x[i] += (x - self.x[i]) / np.sum(np.square(x - self.x[i])) * self.step * np.random.rand()
                else:
                    self.x[i] += self.visual * np.random.uniform(-1, 1)
                self.x[i] = np.clip(self.x[i], self.x_min, self.x_max)
                self.score[i] = self.func(*self.x[i])
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
