# -*- coding: utf-8 -*-
from __future__ import division
import time
from datetime import datetime, timedelta, time
import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytz
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
#import getData

__all__ = ["data","futurepng","days_number","get_vwap","vwap","last","rejectData","getData","get_vabp","get_mabp"]

try:
    data = pd.read_csv("rb1901.csv",index_col=0)
    data.index = pd.to_datetime(data.index)
except:
    import  getData
    getData.main()
    data = pd.read_csv("rb1901.csv", index_col=0)
    data.index = pd.to_datetime(data.index)

days_number = np.load("days_number.npy")
da = [data.iloc[days_number[i]:days_number[i+1]] for i in xrange(len(days_number)-1)]

def get_data():
    try:
        data = pd.read_csv("rb1901.csv", index_col=0)
        data.index = pd.to_datetime(data.index)
    except:
        import getData
        getData.main()
        data = pd.read_csv("rb1901.csv", index_col=0)
        data.index = pd.to_datetime(data.index)
    days_number = np.load("days_number.npy")
    da = [data.iloc[days_number[i]:days_number[i + 1]] for i in xrange(len(days_number) - 1)]

def futurepng(da=da):
    _n = len(da)
    _m = _n//5 + (1 if _n%5>0 else 0)
    fig,axes = plt.subplots(_m,5,sharey=True,sharex=True,figsize = (int(50/6*_m),40))
    _i = 0
    for i in range(_m):
        for j in range(5):
            axes[i,j].plot(da[i*5+j]['last'].values,'y',label = 'last',alpha = 0.3)
            #axes[i,j].set_xticklabels(rotation = 120)
            axes[i,j].plot(da[i*5+j]['askPrc'].values,'g--',label = 'askPrc',alpha = 0.3)
            axes[i,j].plot(da[i*5+j]['bidPrc'].values,'r--',label = 'bidPrc',alpha = 0.3)
            _ = np.ones(len(da[i*5+j]['last']))
            axes[i,j].plot(da[i*5+j]['turnover'].values/da[i*5+j]['volume'].values/10,'b-',label = 'weight-meanpric',alpha=0.5)
            axes[i,j].plot(da[i*5+j]['last'].values.cumsum()/(np.ones(len(da[i*5+j])).cumsum()+0.0),'k--',label = 'mean-last',alpha=0.5)
            axes[i,j].set_title("{}-{}".format(da[i*5+j].index[0].month,da[i*5+j].index[0].day))
            axes[i,j].legend(loc = 'best')
            _i +=1
            if _i>26:
                break
        fig.autofmt_xdate()
    plt.subplots_adjust(left = None, bottom = None, right = None, top = None,wspace=0)
    plt.savefig("future.png")
    plt.show()


def get_vwap(diff=1):
    """
    :param diff:  time interval diff*0.5 s
    :return: vwap of diff*0.5 s interval from opening to closing
    """
    def mydiff(x, n=1):
        y = x.diff(n)
        for i in xrange(n):
            y.iloc[i] = x.iloc[i]
        return y

    vwap = [(mydiff(da[i]['turnover'],diff)/mydiff(da[i]['volume'],diff)/10) for i in xrange(len(da))]
    vwap = pd.concat(vwap).sort_index()
    vwap.fillna(method='ffill',inplace = True)# using the front row of data to fill na
    return vwap

def get_mabp():
    y = (data["askPrc"]+data["bidPrc"])/2
    return y

def get_qwap():
    y = (data["askPrc"]*data["askQty"]+data["bidPrc"]*data["bidQty"])/(data["askQty"]+data["bidQty"])
    return y

vwap = get_vwap()

last = data["last"]

def rejectData(dt=data,min=2):
    """
    reject 2 min data of open/close
    :return: data
    """
    _m = int(min*60*2)
    if _m <= 0:
        return dt[2000:]
    logging.info("start rejectData")
    #drop = np.full((len(days_number)-1)*240,0)
    if not isinstance(dt,pd.core.frame.DataFrame):
        dt = pd.DataFrame(dt)
    dpdata = dt.copy()
    dpdata.loc[:, 'index'] = (np.ones(len(dpdata), dtype='int').cumsum() - 1)
    dpdata.set_index('index', inplace=True)
    _end = np.arange(days_number[-1]-_m,days_number[-1])
    _begin = np.arange(days_number[0],days_number[0]+_m)
    _tmpindex = np.r_[[np.arange(_i,_i+_m) for _i in days_number[:-1]]].reshape(-1)
    _index = np.r_[_begin,_tmpindex,_end]
    logging.debug(_index)
    try:
        dpdata.drop(index=_index, inplace=True)
    except:
        logging.debug("reject {} time fail ".format(min))

    return dpdata
    #d_test_index = np.random.choice(d_train.index, int(len(d_train.index) * .2)).sort()

    #d_test_index = np.random.choice(np.array(d_train.index), size=int(len(d_train.index) * .2), replace=False)
    # d_test_index.sort()
    # d_test = d_train.loc[d_test_index]
    # try:
    #     d_train.drop(index=d_test_index, inplace=True)
    # except:
    #     print "already deleted"

    # 构造测试集合与训练集合，训练集合80%的数据量，测试集合20%的数据量

if __name__ == "__main__":
    #plt.plot(get_vwap())
    #plt.show()
    print rejectData(data,0)["2018-09"]
