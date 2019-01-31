# -*- coding: utf-8 -*-
from __future__ import division

import logging
import re
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn import linear_model, kernel_ridge
import xgboost as xgb

import briefstats

def qqplot( data, dist= 'norm'):
    """

    :type dist: str,'norm'
    """
    if hasattr(data, "values"):
        data = data.values
    if not hasattr(data, "reshape"):
        data = np.array(data)
    data = data.reshape(-1)
    data[np.isnan(data)] = np.nanmean(data)
    stats.probplot(data, dist = dist, plot = plt)
    plt.show()


def r_square(y, pre_y, df=1):
    """
    :param y: real data
    :param pre_y: predicted data
    :param df: rank
    :return: r2 and adjR2
     SST = sum((y-mean(y))**2)
     SSE = sum((y-pre_y)**2)
     R2 = 1 - SSE/SST
    """
    _y = get_values(y)
    _pre = get_values(pre_y)
    _n = len(y)
    SST = np.square(y-y.mean()).sum()
    _pre = _pre.reshape(-1)
    _y = _y.reshape(-1)
    SSE = np.square(_y-_pre).sum()
    R2 = 1-SSE/SST
    adjR2 = 1 - (1-R2)*(_n-1)/(_n-1-df)
    return {"R2":round(R2,4),"adjR2":round(adjR2,4),'MSE':SSE/len(y)}

def get_res(y,pre_y,df=1):
    _y = get_values(y)
    _pre = get_values(pre_y)
    _n = len(y)
    _pre = _pre.reshape(-1)
    _y = _y.reshape(-1)
    SSE = np.square(_y - _pre).sum()
    return SSE/(_n-df)


def get_values(x):
    """
    :param x:
    :return:
    """
    if hasattr(x,"values"):
        x = x.values
    else:
        if not isinstance(x,np.ndarray):
            x = np.array(x)
    return x

def OLS(X,y,norm = False):
    """
    :param X:
    :param y:
    :return:
    """
    # X = get_values(X)
    # if len(X.shape) == 1:
    #     X = X.reshape(-1,1)
    # y = get_values(y)
    # if len(y.shape) == 1:
    #     y = y.reshape(-1,1)
    # print 1
    logging.debug(["OLS X nan",pd.isna(X).sum()])
    logging.debug(["OLS y nan", pd.isna(y).sum()])
    #print len(X),len(y)
    lr1 = sm.regression.linear_model.OLS(endog=y,exog=sm.add_constant(X))
    # print 2
    lr = sm.regression.linear_model.OLS(endog=y,exog=X)
    res1 = lr1.fit()
    res = lr.fit()
    return {"1": res1,
            "-1": res,
            "type": 'OLS'}
def get_kernel_coef(v):
    try:
        alpha = v["alpha"]
    except:
        alpha = 10 ** 0.5
    try:
        kernel = v["kernel"]
    except:
        kernel = 'rbf'
    try:
        gamma = v["gamma"]
    except:
        gamma = 1
    try:
        degree = v["degree"]
    except:
        degree = 3
    try:
        coef0 = v["coef0"]
    except:
        coef0 = 1
    return alpha,kernel,gamma,degree,coef0
def ge_summary(ans):
    return pd.DataFrame(ans, index=ans["index"],columns=["coef", "std", "t", "p_values", "left", "right"])

def beta_test(X,beta,y,alpha=0.05):
    t_test = {0.05:1.96,0.1:1.64,0.01:2.58}
    col = X.columns
    X = get_values(X)
    y = get_values(y).reshape(-1)
    beta = get_values(beta).reshape(-1)
    pre_y = X.dot(beta)
    n = len(X)
    p = len(X[0])
    res = get_res(y, pre_y, p)
    try:
        H = np.linalg.inv((X.T).dot(X))
        Hii = np.array([H[i, i] for i in xrange(len(H))])
    except:
        Hii = float('inf')
    #Hii = np.array([H[i,i] for i in xrange(len(H))])
    std = np.sqrt(res*Hii).reshape(-1)
    t_beta = beta/std
    interval = std*1.96
    _left = beta - interval
    _right = beta + interval
    p_values = 2*(1-stats.t.cdf(np.abs(t_beta),n-p))
    r2 =r_square(y,pre_y,p)
    return {"coef": beta,
           "index":col,
            "p_values": np.round(p_values,3),
            "std": np.round(std,2),
            "t": np.round(t_beta,2),
            "left": np.round(_left,2),
            "right": np.round(_right,2),
            "r2":r2}
def get_alpha(v,default=1):
    try:
        alpha = v["alpha"]
    except:
        alpha = default
    if alpha is None:
        alpha = default
    return alpha

def GLS(X,y,model = "ols",**kwargs):
    """
    model = "ols","ridge","lasso","lar"
    """
    if model == "ols":
        md1 = linear_model.LinearRegression(fit_intercept=True).fit(X,y)
        md0 = linear_model.LinearRegression(fit_intercept=False).fit(X,y)
    if model == "ridge":
        alpha = get_alpha(kwargs,default=10**0.5)
        md1 = linear_model.Ridge(alpha=alpha,fit_intercept=True).fit(X,y)
        md0 = linear_model.Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
    if model == 'lasso':
        alpha = get_alpha(kwargs, default=0.1)
        md1 = linear_model.Lasso(alpha=alpha,fit_intercept=True).fit(X,y)
        md0 = linear_model.Lasso(alpha=alpha, fit_intercept=False).fit(X, y)
    if model == 'lar':
        """
        TO DO
        """
        md1 = linear_model.Lars(fit_intercept=True).fit(X,y)
        md0 = linear_model.Lars(fit_intercept=False).fit(X,y)
    if model == 'kernel':
        alpha, kernel, gamma, degree, coef0 = get_kernel_coef(kwargs["alpha"])
        md1 = kernel_ridge.KernelRidge(alpha=alpha,kernel=kernel,gamma=gamma,degree=degree,coef0=coef0).fit(X,y)
        md0 = md1
    if model == 'xgb':
        md1 = xgb.XGBRegressor().fit(X,y)
        md0 = md1
    return {"1":md1,
            "-1":md0,
            "type":'GLS'}

def OLS_test(tX,ty,res,reportY = False):
    """
    :param tX:   test_X
    :param ty:   test_y
    :param res: model
    :return:       -1, 1
                R2  *  *
            -adjR2  *  *
    """
    df = len(tX.columns)
    #print res
    preY = res["-1"].predict(tX)
    if res["type"] == "OLS":
        preY1 = res["1"].predict(sm.add_constant(tX))
    else:
        preY1 = res["1"].predict(tX)
    if reportY:
        return {"1": r_square(ty,preY1,df),
                "-1": r_square(ty,preY,df),
                "predY": preY,
                "predY1": preY1}
    else:
        return {"1": r_square(ty, preY1, df),
                "-1": r_square(ty, preY, df)}


def get_y(yType = "last",ts=20,rejTime=0,is_log = False,isDiff = True ):
    """
    :param yType "last","vwap","mabp","vabp"
    :param ts: tagat + ts s
    :param rejTime: reject open/close time min
    :param is_log:
    :return:
    """
    #print (yType)
    if yType == "vwap":
        y = briefstats.get_vwap(1)
        y.rename("vwap",inplace=True)
    if yType == "last":
        y = briefstats.data["last"]
        y.rename("last", inplace=True)
    if yType == "mabp":
        y = briefstats.get_mabp()
        y.rename("mabp", inplace=True)
    if yType == "qwap":
        y = briefstats.get_qwap()
        y.rename("qwap", inplace=True)
    if is_log:
        y = np.log(y)
    _ts = ts
    if isDiff:
        y = (y.diff(_ts)).shift(-_ts)
    else:
        y = y.shift(-ts)
    if rejTime>0:
        y = briefstats.rejectData(y,rejTime)
    else:
        logging.info("[:-{}]".format(_ts))
    return y[:-_ts]
def orth(x,y):
    if not isinstance(y,np.ndarray):
        y = np.asarray(y).reshape(-1)
    z = x - (x.dot(y))*y/(y.dot(y))
    return z


def get_args(v, a,default=1,lb=None):
    """
    get list_like args
    """
    try:
        x = a[v]
    except:
        x = a
    if x is None or isinstance(x,dict):
        x =default
    try:
        x[0]
        if not isinstance(x,np.ndarray):
            x = np.array(x)
        if lb is not None:
            x = x[x>=lb]
    except:
        x = [x,]
    return x


def get_rsv(y, window=10):
    _window = window
    _stats = y.rolling(window=_window)
    H = _stats.max()
    H[:_window] = y[:_window].cummax()
    L = _stats.min()
    L[:_window] = y[:_window].cummin()
    tmp = (y.values - L.values) / (H.values - L.values + (H.values == L.values))
    return pd.Series(tmp)

def get_hash(varName=None,args=None):
    varName = list(set(varName+args.keys()))
    def get_varName(varName,args):
        ans=[]
        for val in varName:
            try:
                tmp = [str(val)+'{}'.format(_args) for _args in args[val]]
            except:
                tmp = [str(val),]
            ans +=tmp
        return ans
    allvar = get_varName(varName,args)
    allvar = "-".join(sorted(allvar))
    return hashlib.md5(allvar).hexdigest()

def get_Data(**kw):
    kwargs = {"varName":None,
            "args":None,
             "qcut":0,
              "qType":"norm",
              "orth":False
              }
    #kwargs["orth"] = True
    kwargs.update(kw)
    data = briefstats.data
    data.loc[:,"vwap"] = briefstats.get_vwap(1).values.reshape(-1)
    X = pd.DataFrame()
    _varName = kwargs["varName"]
    if not isinstance(_varName,list):
        _varName = [_varName,]
    filename = get_hash(_varName, kwargs["args"])
    try:
        __col = np.load('data/col{}.npy'.format(filename))
        __index = np.load("data/index{}.npy".format(filename))
        __values = np.load("data/values{}.npy".format(filename))
        X = pd.DataFrame(__values,columns=__col,index=__index)
        return X
    except:
        pass
    mabp = (data["askPrc"] + data["bidPrc"]) / 2
    mabpD = mabp.diff(1)
    mabpD.iloc[0] = 0
    _ratio = ((data["askQty"] * data["askPrc"] - data["bidQty"] * data["bidPrc"]) / (data["askQty"] * data["askPrc"] + data["bidQty"] * data["bidPrc"])).values.reshape(-1)

    qwap = (data["askPrc"]*data["askQty"] + data["bidPrc"]*data["bidQty"]) / (data["askQty"]+data["bidQty"])
    qwapD = qwap.diff(1)
    qwapD.iloc[0] = 0

    def args(v,default=1,lb=None):
        return get_args(v,kwargs['args'],default,lb=lb)

    for varName  in _varName:
        try:
            print -1,varName
            X.loc[:,varName] = data[varName].values.reshape(-1)
            #logging.debug(tmp)
            continue
            # return tmp
        except:
            try:
                _vwap = re.match("vwap",varName).span()
                if _vwap is not None:
                    tmp = varName[_vwap[1]:]
                    if tmp == "D":
                        X.loc[:,"vwapD"] = data["vwap"].diff(1).values.reshape(-1)
                        X.iloc[0, -1] = 0
                        continue
                        # return X
                    if tmp == "Log":
                        X.loc[:,"vwapLog"] = np.log(data["vwap"]).diff(1).values.reshape(-1)
                        X.iloc[0, -1] = 0
                        continue
                        # return X
                    if tmp == "":
                        X.loc[:,"vwap"] = briefstats.get_vwap(1).values.reshape(-1)
                        continue
                    if tmp == "DEWM":
                        vwap = data["vwap"].diff(1)
                        vwap[0] = 0
                        vwapEwm = vwap.ewm(com=1).mean()
                        X.loc[:,"vwapDEWM"] = vwapEwm.values
                        # return X
            except:
                logging.debug("X data do not have vwap")
            try:
                _last = re.match("last", varName).span()
                if _last is not None:
                    logging.debug(["_last is not none",_last])
                    tmp = varName[_last[1]:]
                    logging.debug(tmp)
                    if tmp == "D":
                        X.loc[:, "lastD"] = data["last"].diff(1).values.reshape(-1)
                        X.iloc[0, -1] = 0
                        continue
                        # return X
                    if tmp == "Log":
                        X.loc[:, "lastLog"] = np.log(data["last"]).diff(1).values.reshape(-1)
                        X.iloc[0, -1] = 0
                        continue
                        # return X
            except:
                logging.debug("X data do not have last")
            ####  mabp
            if varName == "mabp":
                try:
                    X.loc[:,"mabp"] = mabp.values.reshape(-1)
                except:
                    X.loc[:, "mabp"] = mabp.reshape(-1)
                continue
            if varName ==  "mabpEWM":
                for _com in args(varName,1):
                    X.loc[:,"mabpEWM{}".format(_com)] = pd.DataFrame(mabp).ewm(com=_com).mean().values.reshape(-1)
                continue
            if varName == "mabpD":
                for _window in args(varName,1,lb=1):
                    mabpDw = mabpD.rolling(window=_window).sum()
                    mabpDw.iloc[:_window] = mabpD.values.reshape(-1)[:_window].cumsum()
                    X.loc[:,"mabpD{}".format(_window)] = mabpDw.values.reshape(-1)
                #print X
                #logging.error(X)
                continue
            if varName == "mabpDEWM":
                for _com in args(varName, 1):
                    mabpDw = mabpD.ewm(com=_com).mean()
                    X.loc[:,"mabpDEWM{}".format(_com)] = mabpDw.values.reshape(-1)
                continue
            #### qwap
            if varName == "qwap":
                X.loc[:,"qwap"] = qwap.values.reshape(-1)
                continue
            if varName == "qwapD":
                X.loc[:,"qwapD"] = qwapD.values.reshape(-1)
                continue
            if varName == "qwapEWM":
                tmp = qwap.ewm(com=1).mean().values.reshape(-1)
                X.loc[:,"qwapDEWM"] =  tmp
                continue
            if varName == "qwapDEWM":
                tmp = qwapD.ewm(com=1).mean().values.reshape(-1)
                X.loc[:, "qwapDEWM"] = tmp
                continue

            if varName == "askDaskbidQty":
                X.loc[:, "askDaskbidQty"] = data["askQty"].values / (data["bidQty"].values + data["askQty"].values)
                continue
                # return X
            if varName == "askDaskbidQtyEWM":
                for _com in args(varName,1):
                    askQty = data["askQty"].ewm(com=_com).mean()
                    bidQty = data["bidQty"].ewm(com=_com).mean()
                    X.loc[:, "askDaskbidQty{}".format(_com) ] = askQty.values / (bidQty.values + askQty.values)
                continue
                # return X
            if varName == "askDaskbidQtyR":
                for _window in args(varName,2,lb=1):
                    askQty = data["askQty"].rolling(window=_window).sum()
                    askQty[:_window] = data["askQty"].values[:_window].cumsum()
                    bidQty = data["bidQty"].rolling(window=_window).sum()
                    bidQty[:_window] = data["bidQty"].values[:_window].cumsum()
                    X.loc[:,"askDaskbidQtyR{}".format(_window)] = (askQty.values/(bidQty.values + askQty.values)).reshape(-1)
            # if varName == "ratio":
            #     openInterestD = data["openInterest"].diff(1)
            #     openInterestD.iloc[0] = 0
            #     X.loc[:, "ratio"] = (openInterestD.values / (data["volumeD"].values + (data["volumeD"].values == 0))).reshape(-1)
            #     continue
            #     # return X
            # if varName == "ratioL":
            #     openInterestD = data["openInterest"].diff(1)
            #     openInterestD.iloc[0] = 0
            #     ratio = openInterestD.values/(data["volumeD"].values+(data["volumeD"].values==0))
            #     logging.error(pd.isna(ratio).sum())
            #     X.loc[:,"ratioL"] = (pd.cut(ratio, bins=[-1.1, -0.75, -0.25, 0.25, 0.75, 1], labels=False) - 3).reshape(-1)
            #     logging.error(pd.isna(X).sum())
            #     continue
            if varName == "fundSpread":
                fundSpread = data["askQty"]*data["askPrc"]-data["bidQty"]*data["bidPrc"]
                for _window in args(varName,1,lb=1):
                    tmp = fundSpread.rolling(window=_window).sum()
                    tmp[:_window] = fundSpread[:_window].cumsum()
                    X.loc[:,"fundSpread{}".format(_window)] = tmp.values.reshape(-1)
                continue
            if varName == "fundSpreadEWM":
                for _com in args(varName,1):
                    ask = pd.DataFrame(data["askQty"] * data["askPrc"]).ewm(com=_com).mean()
                    bid = pd.DataFrame(data["bidQty"] * data["bidPrc"]).ewm(com=_com).mean()
                    X.loc[:, "fundSpreadEWM{}".format(_com)] = (ask - bid).values.reshape(-1)
                continue
            if varName == "askbidDtotalRatio":
                X.loc[:,"askbidDtotalRatio"] = ((data["askQty"]*data["askPrc"]-data["bidQty"]*data["bidPrc"])/(data["askQty"]*data["askPrc"]+data["bidQty"]*data["bidPrc"])).values.reshape(-1)
                continue
            if varName == "askbidDtotalRatioR":
                ask = data["askQty"] * data["askPrc"]
                bid = data["bidQty"] * data["bidPrc"]
                for _window in args(varName,8,lb=1):
                    _window = int(_window)
                    tmpask = ask.rolling(window= _window).sum()
                    tmpask[:_window] = ask[:_window].cumsum()
                    tmpbid = bid.rolling(window=_window).sum()
                    tmpbid[:_window] = bid[:_window].cumsum()
                    X.loc[:,"askbidDtotalRatioR{}".format(_window)]= ((tmpask.values - tmpbid.values)/(tmpask.values + tmpbid.values)).reshape(-1)
                continue

            if varName == "askbidDtotalRatioEWM":
                for com in args(varName,0.1):
                    #com = com/10
                    ask = data["askQty"] * data["askPrc"]
                    ask = ask.ewm(com=com).mean()
                    bid = data["bidQty"] * data["bidPrc"]
                    bid = bid.ewm(com=com).mean()
                    X.loc[:, "askbidDtotalRatioEWM{}".format(com)] = (
                                (ask.values - bid.values) / (ask.values + bid.values)).reshape(-1)
                continue
            # if varName == "askbidDturnover": #wuxiao
            #     D = np.array(map(lambda x: 1 if x > 0 else (0 if x == 0 else -1), mabpD.values))
            #     tmp = ((data["bidQty"]*data["bidPrc"]+data["askQty"]*data["askPrc"])/data["turnoverD"]*10)*D
            #     tmp[np.isinf(tmp)] = np.nan
            #     tmp.fillna(0,inplace=True)
            #     X.loc[:,"askbidDturnover"] = tmp.values.reshape(-1)
            #     #print X
            #     logging.debug(["x nan",pd.isna(X["askbidDturnover"]).sum()])
            #     continue

            if varName == "sov":
                D = np.array(map(lambda x: 1 if x > 0 else (0 if x == 0 else -1), mabpD.values))
                for _window in args(varName,1,lb=1):
                    obv = pd.Series(D * data["volumeD"]).rolling(window=_window).sum()
                    obv[:_window] = (D * data["volumeD"])[:_window].cumsum()
                    logging.debug(obv.values.reshape(-1))
                    X.loc[:,"sov{}".format(_window)] = obv.values.reshape(-1)
                    logging.debug(X)
                continue
            if varName == "sovEWM":
                D = np.array(map(lambda x: 1 if x > 0 else (0 if x == 0 else -1), mabpD.values))
                for _com in args(varName, 1):
                    obvD = pd.DataFrame(D * data["volumeD"]).ewm(com = _com).mean()
                    X.loc[:, "sovEWM{}".format(_com)] = obvD.values.reshape(-1)
                    logging.debug(obvD.values.reshape(-1))
                continue
            if varName == "soo":
                D = np.array(map(lambda x: 1 if x != 0 else 0, mabpD.values))
                openInterestD = data["openInterest"].diff(1)
                for _window in args(varName, 60):
                    openInterestD.iloc[0] = 0
                    soo = pd.DataFrame(D * openInterestD).rolling(window=_window).sum()
                    soo.iloc[:_window,0] = (D * openInterestD).values[:_window].cumsum().reshape(-1)
                    X.loc[:, "soo{}".format(_window)] = soo.values.reshape(-1)
                continue
            if varName == "sooEWM":
                D = np.array(map(lambda x: 1 if x != 0 else 0, mabpD.values))
                openInterestD = data["openInterest"].diff(1)
                openInterestD.iloc[0] = 0
                for _com in args(varName, 1):
                    oboD = pd.DataFrame(D * openInterestD).ewm(com = _com).mean()
                    X.loc[:, "sooEWM{}".format(_com)] = oboD.values.reshape(-1)
                continue
            if varName == "signUpDown":
                D = np.array(map(lambda x: 1 if x > 0 else (0 if x == 0 else -1), mabpD.values))
                for _window in args(varName,2,lb=1):
                    tmp = pd.DataFrame(D).rolling(window=_window).sum()
                    tmp.iloc[:_window,0] = D[:_window].cumsum()
                    if True:
                        tmp = pd.qcut(tmp.values.reshape(-1), 10, duplicates='drop', labels=False)
                        tmp = tmp * (10 / tmp.max())
                        X.loc[:, "signUpDown{}".format(_window)] = tmp.reshape(-1)
                    else:
                        X.loc[:,"signUpDown{}".format(_window)] = tmp.values.reshape(-1)
                continue
            if varName == "signUpDownL":
                D = np.array(map(lambda x: 1 if x > 0 else (0 if x == 0 else -1), mabpD.values))
                tmp = np.full(shape=(len(D),),fill_value=0.0)
                _t = 0
                for i in xrange(len(tmp)):
                    tmp[i] = _t
                    if D[i] > 0:
                        if _t > 0:
                            _t += 1
                        else:
                            _t = 1
                    elif D[i] < 0:
                        if _t < 0:
                            _t -= 1
                        else:
                            _t = -1
                    else:
                        _t = 0
                X.loc[:,"signUpDownL"] = tmp.reshape(-1)
                continue
            if varName == "midDvwap":
                for _diff in args(varName,5,lb=1):
                    vwap = briefstats.get_vwap(_diff)
                    X.loc[:,"midDvwap{}".format(_diff)] = mabp.values - vwap.values
                    continue
            if varName == "qwapDvwap":
                qwap = briefstats.get_qwap()
                for _diff in args(varName,5,lb=1):
                    vwap = briefstats.get_vwap(_diff)

                    X.loc[:, "qwapDvwap{}".format(_diff)] = qwap.values - vwap.values
                #print X
                continue

            if varName == "rsv":
                for _window in args(varName,8,lb=2):
                    X.loc[:,'rsv{}'.format(_window)]=get_rsv(mabp,window=_window).values.reshape(-1)
                continue
            if varName == "rsvEWM":
                for _com in args(varName,0.5):
                    rsv = get_rsv(mabp)
                    rsvEWM = rsv.ewm(_com)
                    X.loc[:,"rsvEWM{}".format(_com)] = rsvEWM.mean()
                continue
            if varName == "rsvEWM":
                for _com in args():
                    pass
    if kwargs["qcut"]>0:
        _columns = X.columns
        if kwargs["qType"] == 'rank':
            for _col in _columns:
                tmp = pd.qcut(X.loc[:,_col],kwargs["qcut"],duplicates='drop',labels=False)
                tmp = tmp*(kwargs["qcut"]/tmp.max())
                X.loc[:,_col] = tmp.values.reshape(-1)
        elif kwargs["qType"] in {'mid','left','right'}:
            for _col in _columns:
                tmp = pd.qcut(X.loc[:,_col],kwargs["qcut"],duplicates='drop').apply(lambda x: getattr(x, kwargs["qType"])).pipe(np.asarray)
                X.loc[:,_col] = tmp.reshape(-1)
        else:
            pass
    if kwargs["orth"]:
        _columns = X.columns
        for _col in _columns:
            tmp = orth(X.loc[:,_col],_ratio)
            try:
                X.loc[:, _col] = tmp.values.reshape(-1)
            except:
                X.loc[:, _col] = tmp.reshape(-1)

    filename = get_hash(_varName, kwargs["args"])
    try:
        np.save('data/col{}.npy'.format(filename),X.columns)
        np.save("data/index{}.npy".format(filename),X.index)
        np.save("data/values{}.npy".format(filename),X.values)
        logging.debug("save sucessed")
    except:
        logging.debug("save failed")
        pass
    return X