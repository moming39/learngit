# -*- coding: utf-8 -*-
from multiprocessing import Pool
import numpy as np
def f(x,y):
    #return x+y
    return lambda z=0:x+y
def mapf(x):
    return f(*x)

myPool = Pool(10)

a = np.ones(shape=(10,2))

res = myPool.map(mapf,a)

print res

