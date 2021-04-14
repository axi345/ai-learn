此为ailearn人工智能算法包。包含了Swarm、RL、nn、utils四个模块。

- Swarm模块当中，实现了粒子群算法、人工鱼群算法、萤火虫算法和进化策略，以及一些对智能算法进行评估的常用待优化的函数。
- RL模块包括两部分，TabularRL部分和Environment部分。TabularRL部分集成了一些经典的强化学习算法，包括Q学习、Q(Lambda)、Sarsa、Sarsa(lambda)、Dyna-Q等。Environment部分集成了一些强化学习经典的测试环境，如FrozenLake问题、CliffWalking问题、GridWorld问题等。
- nn模块包括一些常用的激活函数及损失函数。
- utils模块包括一些常用的功能，包括距离度量、评估函数、PCA算法、标签值与one-hot编码的相互转换、Friedman检测等等。

安装方式（在终端中输入）：

```shell
pip install ailearn
```

更新方式（在终端中输入）：

```shell
pip install ailearn --upgrade
```

更新历史：

- 2018.4.10   0.1.3   第一个版本，首次实现了粒子群算法和人工鱼群算法，首次集成到pip当中。
- 2018.4.16   0.1.4   加入了进化策略的实现，添加了Evaluation模块。
- 2018.4.18   0.1.5   添加了TabularRL模块和Environment模块。
- 2018.4.19   0.1.8   将TabularRL模块和Environment模块整合为RL模块，添加了项目的相关描述,更新了相关协议
- 2018.4.25   0.1.9   输出信息由中文改为英文，并更新了一些已知错误
- 2019.1.15   0.2.0   添加了utils模块，加入了一些常用的功能，包括距离度量、评估函数、PCA算法、标签值与one-hot编码的相互转换、Friedman检测等等；添加了nn模块，加入了一些常用的激活函数及损失函数；更新了Swarm模块的算法，使它们更新得更快。
- 2021.4.6    0.2.1   添加了爬虫工具，增加了RL模块与Swarm模块的示例；添加强化学习经典环境Windy GridWorld环境。



其他更新：

-



项目网址：

https://pypi.org/project/ailearn/

https://github.com/axi345/ailearn/
