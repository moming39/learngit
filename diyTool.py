# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def varname(p):
    """
    :param    p
    :return: "p"
    """
    import inspect,re
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)
def myrename(p):
    import inspect, re
    _name = None
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bmyrename\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            _name = m.group(1)
            break
    p.rename(_name,inplace= True)
    return p
def allplot(var,begin = None,end=None,size=(20,16)):
    if not isinstance(var,list):
        var = [var,]
    plt.figure(figsize=size)
    for _var in var:
        if hasattr(_var,"values"):
            _var = var.values
        plt.plot(_var[begin:end],label = _var.name)