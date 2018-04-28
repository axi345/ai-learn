# -*- coding: utf-8 -*-
from Swarm import PSO
from ailearn.Evaluation import Ackley

f=Ackley()
p=PSO(f.func,param_len=50,x_min=f.min,x_max=f.max)
p.solve(50,verbose=True)
