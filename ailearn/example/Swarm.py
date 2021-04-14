# Using PSO for optimization
from ailearn.Swarm import PSO


def func(x, y):
    return -(x ** 2 + y ** 2)


p = PSO(func=func, param_len=2, x_min=[-5, -5], x_max=[5, 5])
x, y = p.solve()
print(x, y)
