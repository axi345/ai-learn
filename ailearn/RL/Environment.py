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


# CliffWalking环境
class CliffWalking:
    def __init__(self):
        self.state = 0
        self.n_actions = 4
        self.n_states = 48

    def reset(self):
        self.state = 0
        return self.state

    def step(self, a):
        if a == 0:  # 往左走
            if not (self.state % 12 == 0):
                self.state -= 1
        elif a == 1:  # 往上走
            if not (self.state > 35):
                self.state += 12
        elif a == 2:  # 向右走
            if not ((self.state + 1) % 12 == 0):
                self.state += 1
        else:  # 往下走
            if not (self.state < 12):
                self.state -= 12
        if 0 < self.state < 11:
            reward = -100
        else:
            reward = -1
        if 0 < self.state < 12:
            done = True
        else:
            done = False
        return self.state, reward, done, None


# FrozenLake环境
class FrozenLake:
    def __init__(self, n):
        self.state = 0
        self.n = n
        self.n_actions = 4
        self.n_states = n * n

    def reset(self):
        self.state = 0
        return self.state

    def step(self, a):
        # 执行动作
        if a == 0:  # 往左走
            if not (self.state % self.n == 0):
                self.state -= 1
        elif a == 1:  # 往上走
            if not (self.state < self.n):
                self.state -= self.n
        elif a == 2:  # 向右走
            if not ((self.state + 1) % self.n == 0):
                self.state += 1
        else:  # 向下走
            if not (self.state + 1 > self.n * (self.n - 1)):
                self.state += self.n
        # 其他操作
        if self.n == 4:
            if self.state == 5 or self.state == 7 or self.state == 11 or self.state == 12:
                reward = -10
                done = True
            elif self.state == 15:
                reward = 10
                done = True
            else:
                reward = -1
                done = False
            return self.state, reward, done, None
        elif self.n == 5:
            if self.state == 6 or self.state == 8 or self.state == 10 or self.state == 19 or self.state == 22:
                reward = -15
                done = True
            elif self.state == 24:
                reward = 15
                done = True
            else:
                reward = -1
                done = False
            return self.state, reward, done, None
        elif self.n == 6:
            if self.state == 13 or self.state == 14 or self.state == 16 or self.state == 25 or self.state == 27 or self.state == 29 or self.state == 30:
                reward = -20
                done = True
            elif self.state == 35:
                reward = 20
                done = True
            else:
                reward = -1
                done = False
            return self.state, reward, done, None
        elif self.n == 7:
            if self.state == 16 or self.state == 20 or self.state == 21 or self.state == 25 or self.state == 29 or self.state == 32 or self.state == 39 or self.state == 41 or self.state == 44:
                reward = -25
                done = True
            elif self.state == 48:
                reward = 25
                done = True
            else:
                reward = -1
                done = False
            return self.state, reward, done, None
        elif self.n == 8:
            if self.state == 16 or self.state == 25 or self.state == 27 or self.state == 31 or self.state == 37 or self.state == 38 or self.state == 42 or self.state == 44 or self.state == 54 or self.state == 61:
                reward = -30
                done = True
            elif self.state == 63:
                reward = 30
                done = True
            else:
                reward = -1
                done = False
            return self.state, reward, done, None
        elif self.n == 9:
            if self.state == 28 or self.state == 38 or self.state == 40 or self.state == 44 or self.state == 51 or self.state == 52 or self.state == 54 or self.state == 55 or self.state == 57 or self.state == 59 or self.state == 70 or self.state == 78:
                reward = -35
                done = True
            elif self.state == 80:
                reward = 35
                done = True
            else:
                reward = -1
                done = False
            return self.state, reward, done, None


# GridWorld环境
class GridWorld:
    def __init__(self, n):
        self.state = 0
        self.n = n
        self.n_actions = 4
        self.n_states = n * n

    def reset(self):
        self.state = 0
        return self.state

    def step(self, a):
        # 执行动作
        if a == 0:  # 往左走
            if self.state == (self.n - 2) * (self.n + 1) / 2:
                reward = -5
            elif not (self.state % self.n == 0):
                reward = -1
                self.state -= 1
            else:
                reward = -5
        elif a == 1:  # 往上走
            if self.n ** 2 / 2 - 1 < self.state < (self.n + 1) * self.n / 2 - 1 or (
                    self.n + 1) * self.n / 2 < self.state < (self.n + 2) * self.n / 2:
                reward = -5
            elif not (self.state < self.n):
                reward = -1
                self.state -= self.n
            else:
                reward = -5
        elif a == 2:  # 向右走
            if self.state == (self.n - 2) * (self.n + 1) / 2 + 1:
                reward = -5
            elif not ((self.state + 1) % self.n == 0):
                reward = -1
                self.state += 1
            else:
                reward = -5
        else:  # 向下走
            if self.n * (self.n - 4) / 2 - 1 < self.state < self.n * (self.n - 3) / 2 - 1 or (
                    self.n - 3) * self.n / 2 < self.state < (self.n - 2) * self.n / 2:
                reward = -5
            elif not (self.state + 1 > self.n * (self.n - 1)):
                reward = -1
                self.state += self.n
            else:
                reward = -5
        # 其他操作
        done = False
        if self.state == self.n ** 2 - 1:
            reward = self.n ** 2
            done = True
        return self.state, reward, done, None


# WindyGridWorld环境
class WindyGridWorld:
    def __init__(self):
        self.state = 30
        self.n_actions = 4
        self.n_states = 70

    def reset(self):
        self.state = 30
        return self.state

    def step(self, a):
        # 风力处理
        if self.state % 10 == 3 or self.state % 10 == 4 or self.state % 10 == 5 or self.state % 10 == 8:
            self.state += 10
        elif self.state % 10 == 6 or self.state % 10 == 7:
            self.state += 20
        if self.state > 70:
            self.state = 60 + self.state % 10
        # 执行动作
        if a == 0:  # 往左走
            if not self.state % 10 == 0:
                self.state -= 1
        elif a == 1:  # 往上走
            if not self.state >= 60:
                self.state += 10
        elif a == 2:  # 向右走
            if not self.state % 10 == 9:
                self.state += 1
        else:  # 向下走
            if not self.state < 10:
                self.state -= 10
        # 其他操作
        if self.state == 37:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        return self.state, reward, done, None

    def render(self):
        for i in range(6, -1, -1):
            a = []
            for j in range(10):
                a.append('x')
            if i == 3:
                a[0] = 'S'
                a[7] = 'G'
            if self.state // 10 == i:
                a[self.state % 10] = 'o'
            print(' '.join(a))
