# -*- coding: utf-8 -*-
from __future__ import division

from datetime import datetime

import arctic
import numpy as np
import pandas as pd
from arctic.date._daterange import DateRange
from arctic.hooks import register_get_auth_hook
from pandas import Series
from typing import Dict


def setPWD(user=None, passwd=None):
    if user is not None:
        auth = lambda: None
        auth.user = user
        auth.password = passwd
        register_get_auth_hook(lambda *x, **y: auth)
    else:
        register_get_auth_hook(lambda *x, **y: None)


arctic.Arctic("192.168.1.135:19931")
setPWD("liangxuanshuo", "}[;.KLJI*(%^5g6")
client = arctic.Arctic("192.168.1.135:19931")
lib = client['liangxuanshuo.demoFuture']
data = lib.read("rb1901", )
data = lib.read(symbol='rb1901', date_range=DateRange(start='2018-09-10', end='2018-11-13'))
data.sort_index()
# 将数据按每天分类，并将前一天的夜盘并到当天
da = {}  # type: Dict[str, data]
days = []
starttime = pd.to_datetime(range(60), unit='D', origin=pd.Timestamp('2018-09-27'))


def f(x):
    return x.strftime("%m%d")


# f = lambda x: x.strftime("%m%d")
tmp = 0
for _date in starttime:  # type: daytime
    try:
        if not data['2018' + f(_date)].empty:
            if _date.isoweekday() > 5:
                tmp = f(_date)
                continue
            if tmp != 0:
                da[f(_date)] = data['2018' + tmp: '2018' + f(_date)]
                tmp = 0
            else:
                da[f(_date)] = data['2018' + f(_date)]
            days.append(f(_date))
    finally:
        pass
# 将没有交易的时间点补全，
for _data in days:
    index = pd.to_datetime(
        map(lambda x: datetime.fromtimestamp(x.timestamp()) if hasattr(x, 'timestamp') else x, da[_data].index))
    da[_data].index = index

for _days in days:
    index = pd.Series(map(lambda x: x.timestamp()
    if hasattr(x, 'timestamp') else x, da[_days].index))  # 将时间index 变为标准时间戳
    index_diff = index.diff(1)  # type: Series
    ans = []
    lose_data = pd.DataFrame(columns=data.columns)
    for i in xrange(1, len(index)):
        if  0.6 < index_diff[i] < 50:  # 差分之后对间隔大于0.5 进行补全
            _max = int(round(round(index_diff[i], 1) / 0.5, 0))
            _time = index[i] - 8 * 3600
            for j in xrange(_max - 1):
                # print(j,i,_time,_max)
                _time -= 0.5
                lose_data.loc[pd.to_datetime(datetime.fromtimestamp(_time))] = da[_days].iloc[i - 1]
    da[_days] = pd.concat([lose_data, da[_days]]).sort_index(kind="mergesort")

# 对数据按交易时间进行分类，并设置哑变量
for i in xrange(len(days)):
    da[days[i]]['n'] = 0
    # 晚上
    da[days[i]]['m1'] = 0
    # 早上 休息之前
    da[days[i]]['m2'] = 0
    # 早上 休息之后
    da[days[i]]['p'] = 0
    # 下午
    da[days[i]]['time'] = map(lambda x: x.timestamp() if hasattr(x, 'timestamp') else x, da[days[i]].index)
    time = [7200, 16600, 25200]
    _time = da[days[i]].index[-1].timestamp()
    flg1 = da[days[i]]['time'] > _time - time[0]
    da[days[i]].loc[flg1, 'p'] = 1
    flg2 = map(lambda x: x[0] and x[1],
               np.array([da[days[i]]['time'] > _time - time[1], da[days[i]]['time'] < _time - time[0]]).T)
    da[days[i]].loc[flg2, 'm2'] = 1
    flg3 = map(lambda x: x[0] and x[1],
               np.array([da[days[i]]['time'] > _time - time[2], da[days[i]]['time'] < _time - time[1]]).T)
    da[days[i]].loc[flg3, 'm1'] = 1
    flg4 = da[days[i]]['time'] < (_time - time[2])
    da[days[i]].loc[flg4, 'n'] = 1
    da[days[i]].loc[:, 't'] = da[days[i]]['n'] + 2 * da[days[i]]['m1'] + 3 * da[days[i]]['m2'] + 4 * da[days[i]]['p']
    da[days[i]].drop('time', axis=1, inplace=True)

# def set_first_day_of_one_trade_week():
_tmpweek = 55
for i in xrange(len(days)):
    _begin = da[days[i]].index[0].weekday()
    _end = da[days[i]].index[-1].weekday()
    _week = da[days[i]].index[-1].week
    if _end < _begin:
        # is fristDay.
        da[days[i]].loc[:, "beginDay"] = 1
    else:
        da[days[i]].loc[:, "beginDay"] = 0
    da[days[i]].loc[:, "endDay"] = 0
    if _tmpweek < _week:
        da[days[i - 1]].loc[:, "endDay"] = 1
    _tmpweek = _week


def mydiff(x):
    y = x.diff(1)
    y[0] = x[0]
    return y


for i in xrange(len(days)):
    da[days[i]].loc[:, "turnoverD"] = mydiff(da[days[i]]["turnover"])
    da[days[i]].loc[:, "volumeD"] = mydiff(da[days[i]]["volume"])

data = pd.concat([_da for _da in da.values()]).sort_index(kind='mergesort')


def main():
    # with open("data.json", 'w') as f:
    #    json.dump(da, f)
    data.to_csv("rb1901.csv", header=True, index=True)
    days_number = np.array([0] + [len(da[_daytime]) for _daytime in days]).cumsum()
    # 次日分隔符l
    np.save("days_number", days_number)


if __name__ == "__main__":
    print data.loc["2018-09-27 17:07:20":"2018-09-27 17:07:25", ["volume", "turnover"]]
    print data.loc["2018-09-27 17:07:20":"2018-09-27 17:07:25", ["volumeD", "turnoverD"]]
    main()
