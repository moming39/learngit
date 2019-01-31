# -*- coding: utf-8 -*-

from __future__  import division
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as smf
import scipy.stats as stats
import briefstats
from commonTest import get_Data
data = briefstats.data
data.loc[:,"vwap"] = briefstats.get_vwap(1)

def adfTest(x,alpha = 0.01):
    _n = len(x)
    _level = {0.01:"1%",0.1:"10%",0.05:"5%"}
    if alpha not in _level.keys():
        _level[alpha] = "1%"
    if _n >10000:
        _begin= np.random.randint(0,_n-10000)
    else:
        _begin=0
    _test = sm.tsa.adfuller(x[_begin:_begin+100000])
    if _test[0] <= _test[4][_level[alpha]]:
        ans = True
    else:
        ans = False
    return (ans,_test)

def multiTest(x,alpha = 0.01,repeat = 10):
    for i in xrange(repeat):
        if not adfTest(x,alpha)[0]:
            return False
    return True

def diffTest(x,alpha = 0.01,repeat = 10):
    if hasattr(x,"ndim"):
        if x.ndim > 1:
            try:
                x = x.values.reshape(-1)
            except:
                x = x.reshape(-1)
    if not hasattr(x,"diff"):
        x = pd.Series(np.array(x).reshape(-1))
    d = 0
    while(not multiTest(x,alpha,repeat)):
        x = x.diff(1)
        x = x[1:]
        d+=1
    return d

def alphaTest(varName="last"):
    X = get_Data(varName=varName)
    d = diffTest(X.values)
    print varName,"d:",d
    dta = X
    fig = plt.figure(figsize=(20, 16))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(dta, lags=20, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(dta, lags=20, ax=ax2)
    plt.show()


def qqplot(x, y, size=(20, 16), xName=None, yName=None, Nu=10000):
    from scipy.stats import percentileofscore
    from sklearn.linear_model import LinearRegression
    if not isinstance(x, pd.core.frame.DataFrame):
        df_clu = pd.DataFrame(x)
    else:
        df_clu = x
    if not isinstance(x, pd.core.frame.DataFrame):
        df_samp = pd.DataFrame(y)
    else:
        df_samp = y
    if len(x) > Nu:
        _index = np.random.choice(np.arange(len(x)), size=Nu, replace=False)
        _index.sort()
        df_samp = df_samp.iloc[_index]
        df_clu = df_clu.iloc[_index]
    # theoretical quantiles
    # df_samp, df_clu are two dataframes with input data set
    ref = np.asarray(df_clu)
    samp = np.asarray(df_samp)
    if xName is None:
        try:
            ref_id = df_clu.columns[0]
        except:
            ref_id = df_clu.name
    else:
        ref_id = xName
    if yName is None:
        try:
            samp_id = df_samp.columns[0]
        except:
            samp_id = df_samp.name
    else:
        samp_id = yName

    samp_pct_x = np.asarray([percentileofscore(ref, x) for x in samp])
    # sample quantiles
    samp_pct_y = np.asarray([percentileofscore(samp, x) for x in samp])
    # estimated linear regression model
    p = np.polyfit(samp_pct_x, samp_pct_y, 1)
    regr = LinearRegression()
    model_x = samp_pct_x.reshape(len(samp_pct_x), 1)
    model_y = samp_pct_y.reshape(len(samp_pct_y), 1)
    regr.fit(model_x, model_y)
    r2 = regr.score(model_x, model_y)
    # get fit regression line
    if p[1] > 0:
        p_function = "y= %s x + %s, r-square = %s" % (str(p[0]), str(p[1]), str(r2))
    elif p[1] < 0:
        p_function = "y= %s x - %s, r-square = %s" % (str(p[0]), str(-p[1]), str(r2))
    else:
        p_function = "y= %s x, r-square = %s" % (str(p[0]), str(r2))
    print "The fitted linear regression model in Q-Q plot using data from enterprises %s and cluster %s is %s" % (
        str(samp_id), str(ref_id), p_function)

    # plot q-q plot
    plt.figure(figsize=size)
    x_ticks = np.arange(0, 100, 20)
    y_ticks = np.arange(0, 100, 20)
    plt.scatter(x=samp_pct_x, y=samp_pct_y,marker= 'o', color='blue', alpha=1)
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    # add fit regression line
    plt.plot(samp_pct_x, regr.predict(model_x), color='red', linewidth=4 * (size[0] // 10 + 1), alpha=0.7)
    # add 45-degree reference line
    plt.plot([0, 100], [0, 100], linewidth=2 * (size[0] // 10 + 1), alpha=1)
    plt.text(10, 70, p_function)
    plt.xticks(x_ticks, x_ticks)
    plt.yticks(y_ticks, y_ticks)
    plt.xlabel('cluster quantiles - id: %s' % str(ref_id))
    plt.ylabel('sample quantiles - id: %s' % str(samp_id))
    plt.title('%s VS %s Q-Q plot' % (str(ref_id), str(samp_id)))
    plt.show()

def qqplotDist(data,dist = 'norm',size=(20,16),save=None):
    if hasattr(data,"values"):
        data = data.values
    if not hasattr(data,"reshape"):
        data = np.array(data)
    data = data.reshape(-1)
    data[np.isnan(data)] = np.nanmean(data)
    plt.figure(figsize=size)
    stats.probplot(data,dist = dist,plot = plt)
    if save is not None:
        plt.savefig('figure/{}{}.png'.format(save,time.strftime("%m-%d_%H_%M_%S", time.localtime(time.time()))))
    else:
        plt.show()

if __name__ == "__main__":
    _name = ["lastD", "lastLog", "vwapD", "vwapLog"
        , "signUpDownL",
             "sov", "sovEWM", "qwapD", "askDaskbidQty",
             "fundSpread", "askbidDtotalRatio", "fundSpreadEWM", "askDaskbidQtyEWM",
             "mabpD", "mabpDEWM", "soo", "sooD", "signUpDown"]
    ans = {}
    for _na in _name:
        ans[_na] = diffTest(get_Data(varName=_na))
    dt = pd.Series(ans)
    print dt
    dt.to_csv("testAns/alphaTest{}.csv".format(len(_name)), header=True, index=True)