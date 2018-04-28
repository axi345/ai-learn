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
from setuptools import setup, find_packages

long_description = '''
此为ailearn人工智能算法包。包含了Swarm、EAS、Evaluation、RL四个模块。
Swarm模块当中，实现了粒子群算法、人工鱼群算法和萤火虫算法。
EAS模块当中，实现了进化策略。
Evaluation模块集成了一些对智能算法进行评估的常用待优化的函数。
RL模块包括两部分，TabularRL部分和Environment部分。
TabularRL部分集成了一些经典的强化学习算法，包括Q学习、Q(Lambda)、Sarsa、Sarsa(lambda)、Dyna-Q等。
Environment部分集成了一些强化学习经典的测试环境，如FrozenLake问题、CliffWalking问题、GridWorld问题等。

更新历史：
2018.4.10   0.1.3   第一个版本，首次实现了粒子群算法和人工鱼群算法，首次集成到pip当中。
2018.4.16   0.1.4   加入了进化策略的实现，添加了Evaluation模块。
2018.4.18   0.1.5   添加了TabularRL模块和Environment模块。
2018.4.19   0.1.8   将TabularRL模块和Environment模块整合为RL模块，添加了项目的相关描述,更新了相关协议
2018.4.25   0.1.9   输出信息由中文改为英文，并更新了一些已知错误

项目网址：
https://pypi.org/project/ailearn/
https://github.com/axi345/ailearn/
'''

setup(
    name='ailearn',
    version='0.1.9',
    description='A lightweight package for artificial intelligence',
    long_description=long_description,
    author='ZHAO Xingyu; AN Yuexuan',
    author_email='757008724@qq.com',
    license='Apache License, Version 2.0',
    url='http://github.com/axi345/ailearn',
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Environment :: Console',
    ],
    zip_safe=False,
    install_requires=['numpy', 'pandas'],
)
