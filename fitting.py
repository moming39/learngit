# -*- coding: utf-8 -*-
from __future__ import division
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

import briefstats
from multiprocessing import Pool
from commonTest import OLS,GLS,OLS_test,get_y,get_Data,beta_test
from alphaTest import qqplotDist
data = briefstats.data
days_number = briefstats.days_number
vwap = briefstats.get_vwap(1)
last = data['last']
_EXOG = None
_EDOG = None
_DUMMY = ["n","m1","m2","p","t","beginDay","endDay"]
_TRAIN_MEAN = None
_TRAIN_STD = None

def get_train_and_test(X,y,sample = 'c',ratio=0.2,cvK = 10,is_norm = True,outNorm = None):
    """
     X y ratio = len(tX)/len(X), sample = 'r':random,'c':continuous,3,'cv'
    :return: X,y,tX,ty
    """
    if isinstance(X,pd.core.frame.DataFrame):
        d_train = X.copy()
    else:
        d_train = pd.DataFrame(X)
    d_train.loc[:, 'index'] = (np.ones(len(d_train), dtype='int').cumsum() - 1)
    y.index = d_train["index"]
    d_train.set_index('index', inplace=True)
    # d_train = d_train.iloc[:1000]
    # y = y.iloc[:1000]
    if sample == 'c':
        _n = int(len(d_train) * (1 - ratio))
        X = d_train.iloc[:_n]
        ty = y.iloc[_n:]
        tX = d_train.iloc[_n:]
        y = y.iloc[:_n]
        d_train = X
    else:
        np.random.seed(12345)
        _n = len(d_train.index) - cvK*int(len(d_train.index)/cvK * (1-ratio))
        d_test_index = np.random.choice(np.array(d_train.index), size= _n , replace=False)
        d_test_index.sort()
        tX = d_train.iloc[d_test_index].copy()
        logging.debug(["train tX nan", pd.isna(X).sum()])
        ty = y.iloc[d_test_index].copy()

        _flg = np.ones(len(d_train))
        _flg[d_test_index] = 0
        X = d_train.iloc[d_train.index[_flg == 1]].copy()
        y = y.iloc[d_train.index[_flg == 1]].copy()
        # try:
        #     d_train.drop(index=d_test_index, inplace=True)
        #     y.drop(index=d_test_index, inplace=True)
        # except:
        #     logging.error("train test choice failed")


    logging.debug(["train X nan", pd.isna(d_train).sum()])


    if is_norm:
        global _TRAIN_STD,_TRAIN_MEAN,_DUMMY
        _TRAIN_MEAN = X.mean()
        #logging.debug(d_train[0:10])
        #logging.debug(type(d_train))
        _TRAIN_STD = X.std()
        if outNorm == None:
            _col_and_dummy = list(set(_DUMMY) & set(X.columns))
        else:
            logging.debug(outNorm)
            _col_and_dummy = list((set(_DUMMY)|set(outNorm)) & set(X.columns))
        _TRAIN_MEAN[_col_and_dummy] = 0
        _TRAIN_STD[_col_and_dummy] = 1
        for _name in d_train.columns:
            if _TRAIN_STD[_name] == 0:
                _TRAIN_STD[_name] = 1
    else:
        _TRAIN_MEAN = 0
        _TRAIN_STD = 1
    logging.debug([_TRAIN_MEAN, _TRAIN_STD])
    X = (X-_TRAIN_MEAN)/_TRAIN_STD
    tX = (tX-_TRAIN_MEAN)/_TRAIN_STD
    # try:
    #     vX = (vX-_TRAIN_MEAN)/_TRAIN_STD
    #     return X,y,vX,vy,tX,ty
    # except:
    #     logging.info("no verify X")
    if sample == 'cv':
        np.random.seed(1000)
        X_index = X.index.values
        np.random.shuffle(X_index)
        X_index = X_index.reshape(cvK, -1)
        X_index.sort()
        vX = [[X.iloc[~X.index.isin(X_index[i])],X.iloc[X.index.isin(X_index[i])]] for i in xrange(cvK)]
        vy = [[y.iloc[~X.index.isin(X_index[i])],y.iloc[X.index.isin(X_index[i])]] for i in xrange(cvK)]
        logging.info(["k -fold cross validation",cvK])
        return X,y,vX,vy,tX,ty
    return X, y, tX, ty,
    # 构造测试集合与训练集合，训练集合80%的数据量，测试集合30%的数据量

def is_listlike(a):
    try:
        if isinstance(a[0],float) or isinstance(a[0],int):
            return True
        else:
            assert True,"a type Error"
    except:
        return False

def get_y_trans(y,isTrans=True):
    if isTrans:
        logging.debug("y  Trans!")
        return np.sign(y)*np.log(np.abs(y)+1)
    else:
        return y

def get_y_inv(y,isTrans=True):
    if isTrans:
        logging.debug("y  inv Trans!")
        return np.sign(y)*(np.exp(np.abs(y))-1)
    else:
        return y

def main(**kwargs):
    """
    :param kwargs: {"yType":"last",
             "is_log":False,
             "rolling":0,
             "has_log":True,
             "x_col":None,
             "has_time":True,
             "is_norm":True,
             "rejTime":2,
             "x":0,
             "tick_nu":0,
             "ask_bid":0,
             "varTest":False,
             "varName":None,
             "isDiff": True,
             "ts":20,
             "yRes":None,
             "saveyRes":None,
             "args":None,
             "qcut":0,
             "yCut":None,
             "qType":'rank'}
    :return:
    """
    KWARGS ={"yType": "last",
             "is_log": False,
             "has_log": True,
             "has_time": True,
             "is_norm": True,
             "rejTime": 0,
             "varTest": False,
             "varName": None,
             "isDiff": True,
             "ts": 20,
             "yRes": None,
             "saveyRes": None,
             "args": None,
             "qcut": 0,
             "yCut": None,
             "qType": 'rank',
             "GLS": None, #'None:OLS,ols:ols,ridge,lasso,lar,
             'alpha': None,
             'sample': 'cv',
             "cvK":10,
             "isMP":False,
             "isTrans":False}
    KWARGS.update(kwargs)
    if KWARGS["yRes"] is None:

        y = get_y(yType=KWARGS["yType"],is_log=KWARGS["is_log"],ts=KWARGS["ts"],isDiff=KWARGS["isDiff"])
    else:
        y = pd.read_csv("yRes/{}-{}.csv".format(KWARGS["yType"],KWARGS["yRes"]),index_col=0)
    print KWARGS["isTrans"]
    y = get_y_trans(y,KWARGS["isTrans"])
    if KWARGS["qcut"]>0:
        if KWARGS["yCut"] == 'rank':
            tmp = pd.qcut(y, kwargs["qcut"], duplicates='drop', labels=False)
            tmp = tmp * (kwargs["qcut"] / tmp.max())
            y = tmp.values.reshape(-1)
        elif KWARGS["yCut"] in {'mid','left','right'}:
            tmp = pd.qcut(y, kwargs["qcut"], duplicates='drop').apply(lambda x:getattr(x,KWARGS["yCut"])).pipe(np.asarray)
            y = tmp.reshape(-1)
        else:
            pass

    if KWARGS["varTest"]:
        X = get_Data(varName=KWARGS["varName"],qcut = KWARGS["qcut"],qType = KWARGS["qType"],args=KWARGS["args"])[:-KWARGS["ts"]]
        #logging.error(X)
    else:
        print "do not have data"
    logging.debug(["fit X nan", pd.isna(X).sum()])
    logging.debug(["X",len(X),"y",len(y)])
    y = briefstats.rejectData(y,min=KWARGS["rejTime"])
    X = briefstats.rejectData(X,min=KWARGS["rejTime"])
    logging.debug(["reject X nan", pd.isna(X).sum()])
    logging.info("=================X======================")
    logging.info(X.columns)

    logging.info(KWARGS["yType"]+"~~"+"+".join(X.columns))
    # print(3)
    logging.debug(["before train X nan", pd.isna(X).sum()])
    if KWARGS['sample'] == 'cv':
        X,y,vX,vy,tX,ty = get_train_and_test( X, y, is_norm=KWARGS["is_norm"], sample=KWARGS['sample'], cvK = KWARGS["cvK"])
    else:
        X,y,tX,ty = get_train_and_test(X,y,is_norm=KWARGS["is_norm"],sample=KWARGS['sample'])
    # print(4)
    logging.debug(["train X nan", pd.isna(X).sum()])
    global _EDOG,_EXOG
    _EXOG = {"X":X,"tX":tX}
    _EDOG = {"y":y,"ty":ty}

    if KWARGS["GLS"] in {"ols","ridge","lasso","lar","kernel","xgb"}:
        try:
            def mapfunc(param):
                return param["func"](param["X"],param["y"],param["GLS"],alpha=param["alpha"])
            def get_param(alpha):
                param = [{"func":GLS,"X": vX[i][0], "y": vy[i][0], "GLS": KWARGS["GLS"], "alpha": alpha} for i in
                     xrange(KWARGS["cvK"])]
                return param

            if is_listlike(KWARGS["alpha"]):
                if KWARGS["isMP"]:
                    myPool = Pool(10)
                    parm = []
                    for _alpha in KWARGS["alpha"]:
                        parm +=get_param(alpha=_alpha)
                    parm+=[{"func":GLS,"X":X,"y":y, "GLS": KWARGS["GLS"], "alpha":KWARGS['alpha']}]
                    tmpRes = myPool.map(mapfunc,(myGLS,parm))
                    myPool.close()
                    myPool.join()
                    res = tmpRes[-1]
                    resK = [tmpRes[i*KWARGS["cvK"]:(i+1)*KWARGS["cvK"]] for i,_ in enumerate(KWARGS["alpha"])]
                else:
                    resK = [[GLS(vX[i][0],vy[i][0],model=KWARGS["GLS"],alpha=_alpha)
                               for i in xrange(KWARGS["cvK"])] for _alpha in KWARGS["alpha"]]
                    res = GLS(X, y, model=KWARGS["GLS"], alpha=None)
            else:
                if KWARGS["isMP"]:
                    myPool = Pool(10)
                    parm = get_param(alpha=KWARGS['alpha']) + [{"func":GLS,"X":X,"y":y, "GLS": KWARGS["GLS"], "alpha":KWARGS['alpha']}]
                    #print parm
                    print mapfunc(parm[0])
                    tmpRes = myPool.map(mapfunc,parm)
                    myPool.close()
                    myPool.join()
                    resK = tmpRes[:-1]
                    res = tmpRes[-1]
                else:
                    resK = [GLS(vX[i][0],vy[i][0],model=KWARGS["GLS"],alpha=KWARGS['alpha']) for i in xrange(KWARGS["cvK"])]
                    res = GLS(X, y, model=KWARGS["GLS"], alpha=KWARGS['alpha'])
        except:
            logging.debug(["KWARGS[GLS]: error"])
            res = GLS(X,y,model=KWARGS["GLS"],alpha=KWARGS['alpha'])
    else:
        res = OLS(X,y)
    if KWARGS["sample"] == 'cv':
        logging.debug([is_listlike(KWARGS['alpha']),KWARGS['alpha']])
        if not is_listlike(KWARGS["alpha"]):
            cv1R2 = np.zeros(KWARGS["cvK"])
            cv0R2 = np.zeros(KWARGS["cvK"])
            cv1MSE = np.zeros(KWARGS["cvK"])
            cv0MSE = np.zeros(KWARGS["cvK"])
            for i in xrange(KWARGS["cvK"]):
                tmp = OLS_test(vX[i][1],vy[i][1],resK[i])
                cv1R2[i] = tmp["1"]["adjR2"]
                cv0R2[i] = tmp["-1"]["adjR2"]
                cv0MSE[i] = tmp["-1"]["MSE"]
                cv1MSE[i] = tmp["1"]["MSE"]
            cvRes = {"1": [cv1R2.mean(),cv1MSE.mean()],
                     "-1": [cv0R2.mean(),cv0MSE.mean()]
                     }
            res["cvRes"] = cvRes
            logging.debug((["cv cvR2 calculate:",1]))
        else:
            cv1R2 = [0 for _ in KWARGS["alpha"]]
            cv0R2 = [0 for _ in KWARGS["alpha"]]
            cv1MSE = [0 for _ in KWARGS["alpha"]]
            cv0MSE = [0 for _ in KWARGS["alpha"]]
            for i,_alpha in enumerate(KWARGS["alpha"]):
                tmp = [OLS_test(vX[j][1], vy[j][1], resK[i][j]) for j in xrange(KWARGS["cvK"])]
                cv1R2[i] = np.array(map(lambda x:x["1"]["adjR2"],tmp))
                cv0R2[i] = np.array(map(lambda x:x["-1"]["adjR2"],tmp))
                cv0MSE[i] = np.array(map(lambda x:x["1"]["MSE"],tmp))
                cv1MSE[i] = np.array(map(lambda x:x["-1"]["MSE"],tmp))
            cvRes = {"1": [np.array([cv1R2[i].mean() for i in xrange(len(KWARGS["alpha"]))]) , np.array([cv1MSE[i].mean() for i in xrange(len(KWARGS["alpha"]))])],
                     "-1": [np.array([cv0R2[i].mean() for i in xrange(len(KWARGS["alpha"]))]), np.array([cv0MSE[i].mean() for i in xrange(len(KWARGS["alpha"]))])]
                     }
            res["cvRes"] = cvRes
            bestAlpha = cvRes["-1"][0].max()
            tmp = GLS(X, y, model=KWARGS["GLS"], alpha=bestAlpha)
            res["-1"] = tmp["-1"]
            res["1"] = tmp["1"]
            logging.debug((["cv cvR2 calculate:", 2]))
            logging.debug(["res:",res])
    if KWARGS['sample'] == 'cv' and KWARGS["GLS"]!= 'xgb':
        try:
            res["-1test"] = beta_test(tX, res["-1"].coef_, ty)
        except:
            res["-1test"] = beta_test(tX, res["-1"].params, ty)

        try:
            res["1test"] = beta_test(sm.add_constant(tX),res["1"].params,ty)
        except:
            _beta = res["1"].coef_.reshape(-1)
            beta = np.r_[res["1"].intercept_,_beta]
            res["1test"] = beta_test(sm.add_constant(tX),beta,ty)

    if KWARGS["GLS"] == "lasso":
        _col = np.array(X.columns)
        res["-1col"] = _col[np.abs(res["-1"].coef_) > 0]
        try:
            res["1col"] = _col[(np.abs(res["1"].coef_) > 0)[1:]]
        except:
            res["1col"] = _col[(np.abs(res["1"].coef_) > 0)]
    if KWARGS["saveyRes"] is not None:
        dX = pd.concat([X,tX]).sort_index(kind = 'mergesort')
        dY = pd.concat([y,ty]).sort_index(kind = 'mergesort')
        preY = res["1"].predict(sm.add_constant(dX)).values
        logging.debug(["preY",preY])
        dy = pd.DataFrame(dY.values.reshape(-1)-preY.reshape(-1))
        logging.debug(["{}res".format(KWARGS["yType"]),type(dy)])
        logging.debug(dy)
        #dy.rename("{}res".format(KWARGS["yType"]),inplace= True)
        dy.to_csv("yRes/{}-{}.csv".format(KWARGS["yType"],KWARGS["saveyRes"]), header=True, index=True)
        print "yRes/{}-{}.csv".format(KWARGS["yType"],KWARGS["saveyRes"])
        logging.info("yRes/{}-{}.csv".format(KWARGS["yType"],KWARGS["saveyRes"]))

    tmpAns = OLS_test(tX,ty,res,reportY=True)
    testr2 = pd.DataFrame(tmpAns,columns=["-1","1"])
    try:
        qqplotDist(ty-tmpAns["predY"],save="y-preY-QQnorm{}".format(bestAlpha))
        qqplotDist(ty-tmpAns["predY"], save="y-preY-QQnorm{}".format(bestAlpha))
    except:
        qqplotDist(ty - tmpAns["predY"], save="y-preY-QQnorm{}".format(1))
        qqplotDist(ty - tmpAns["predY"], save="y-preY-QQnorm{}".format(1))
    logging.debug(res)
    return res,testr2

def get_exog():
    """
    :return: X,tX
    """
    return _EXOG

def get_edog():
    """
    :return: y,ty
    """
    return _EDOG

def get_norm_args():
    """
    :return: (mean,std)
    """
    return _TRAIN_MEAN, _TRAIN_STD

def loadlogging(filename= None,level = None):
    if filename == None:
        import time
        filename  = "fittingTMP{}.log".format(time.strftime("%Y-%m-%d_%H_%M_%S",time.localtime(time.time())))
    if level == None:
        level = logging.DEBUG
    logging.basicConfig(filename=filename, level=level)

if __name__ == "__main__":
    import logging
    pass