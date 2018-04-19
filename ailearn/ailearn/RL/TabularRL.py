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
import pandas as pd

np.random.seed(1)


class TabularRL(object):
    def __init__(self, n_actions, n_states, learning_rate=0.1, reward_decay=0.9, e_greedy=0.01):
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = np.zeros([n_states, n_actions])

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            a = np.random.choice(self.n_actions)
        else:
            action_list = []
            max = np.max(self.q_table[s, :])
            for i in range(self.n_actions):
                if self.q_table[s, i] == max:
                    action_list.append(i)
            a = np.random.choice(action_list)
        return a

    def learn(self, *args):
        pass


class QLearning(TabularRL):
    def learn(self, s, a, r, s_, done):
        q_predict = self.q_table[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.q_table[s_, :])
        self.q_table[s, a] += self.lr * (q_target - q_predict)


class SarsaLearning(TabularRL):
    def learn(self, s, a, r, s_, a_, done):
        q_predict = self.q_table[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table[s_, a_]
        self.q_table[s, a] += self.lr * (q_target - q_predict)


class SarsaLambda(TabularRL):
    def __init__(self, n_actions, n_states, learning_rate=0.1, reward_decay=0.9, e_greedy=0.1, trace_decay=0.9,
                 style='replacing'):
        super(SarsaLambda, self).__init__(n_actions, n_states, learning_rate, reward_decay, e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()
        self.style = style

    def learn(self, s, a, r, s_, a_, done):
        q_predict = self.q_table[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table[s_, a_]
        error = q_target - q_predict
        assert self.style == 'replacing' or 'accumulating'
        if self.style == 'replacing':
            self.eligibility_trace[s, :] *= 0
            self.eligibility_trace[s, a] = 1
        else:
            self.eligibility_trace[s, a] += 1
        self.q_table += self.lr * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_


class QLambda(TabularRL):
    def __init__(self, n_actions, n_states, learning_rate=0.1, reward_decay=0.9, e_greedy=0.1, trace_decay=0.9):
        super(QLambda, self).__init__(n_actions, n_states, learning_rate, reward_decay, e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def learn(self, s, a, r, s_, done):
        q_predict = self.q_table[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.q_table[s_, :])
        error = q_target - q_predict
        self.eligibility_trace[s, :] *= 0
        self.eligibility_trace[s, a] = 1
        self.q_table += self.lr * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_


class EnvModel(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.database = pd.DataFrame(columns=n_actions, dtype=np.object)

    def store_transition(self, s, a, r, s_, done):
        if s not in self.database.index:
            self.database = self.database.append(
                pd.Series(
                    [None] * len(self.n_actions),
                    index=self.database.columns,
                    name=s
                ))
        self.database.set_value(s, a, (r, s_, done))

    def sample_s_a(self):
        s = np.random.choice(self.database.index)
        a = np.random.choice(self.database.ix[s].dropna().index)  # filter out the None value
        return s, a

    def get_r_s_(self, s, a):
        r, s_, done = self.database.ix[s, a]
        return r, s_, done


class DynaQ(TabularRL):
    def __init__(self, n_actions, n_states, learning_rate=0.1, reward_decay=0.9, e_greedy=0.01, learning_n=5):
        super(DynaQ, self).__init__(n_actions, n_states, learning_rate, reward_decay, e_greedy)
        self.model = EnvModel(list(range(n_actions)))
        self.learning_n = learning_n

    def learn(self, s, a, r, s_, done):
        self.model.store_transition(s, a, r, s_, done)
        q_predict = self.q_table[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.q_table[s_, :])
        self.q_table[s, a] += self.lr * (q_target - q_predict)
        for i in range(self.learning_n):
            s, a = self.model.sample_s_a()
            r, s_, done = self.model.get_r_s_(s, a)
            q_predict = self.q_table[s, a]
            if done:
                q_target = r
            else:
                q_target = r + self.gamma * np.max(self.q_table[s_, :])
            self.q_table[s, a] += self.lr * (q_target - q_predict)
