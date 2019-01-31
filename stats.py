# -*- coding: utf-8 -*-
from __future__ import division
import time
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytz
import sklearn
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
import fitting
from commonTest import OLS_test,OLS,get_y
import briefstats
#data = fitting.clearData()
days_number = fitting.days_number
data = briefstats.data
def get_da(data = data):
    da = [data[days_number[i-1]:days_number[i]] for i in xrange(1,len(days_number))]
    return da
#da = get_da(data)

def test():
    y = get_y(yType="vabp", isDiff=False)
    x = data["openInterest"][:-20]
    x = x*y
    logging.debug([x,y])
    X, y, tX, ty = fitting.get_train_and_test(X=x, y=y)
    res = OLS(X, y)
    testr2 = pd.DataFrame(OLS_test(tX, ty, res))
    return res, testr2

def test2():
    y = get_y(yType="vabp", isDiff=False)
    x = data["openInterest"][:-20]
    #x = x*y
    logging.debug([x,y])
    X, y, tX, ty = fitting.get_train_and_test(X=x, y=y)
    res = OLS(X, y)
    testr2 = pd.DataFrame(OLS_test(tX, ty, res))
    return res, testr2
if __name__ == "__main__":
    import logging
    #_file = "varTestlog/test_{}{}.log".format(varName, time.strftime("%m-%d_%H_%M_%S", time.localtime(time.time())))
    _file = "varTestlog/test_{}{}.log".format("openInterest*vabp",
                                              time.strftime("%m-%d_%H_%M_%S", time.localtime(time.time())))
    logging.basicConfig(filename=_file, level=logging.INFO)
    res,r2= test()
    logging.info("----")
    logging.info(res["1"].summary())
    logging.info(res["-1"].summary())
    logging.info(r2)

    res, r2 = test2()
    logging.info("----")
    logging.info(res["1"].summary())
    logging.info(res["-1"].summary())
    logging.info(r2)