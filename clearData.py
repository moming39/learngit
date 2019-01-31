# -*- coding: utf-8 -*-
from __future__ import division
import time
from datetime import datetime, timedelta, time
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytz
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

import briefstats
import logging

data = briefstats.data
days_number = briefstats.days_number
vwap = briefstats.get_vwap(1)
last = data['last']
_EXOG = None
_EDOG = None
_DUMMY = ["n","m1","m2","p","t","beginDay","endDay"]
_TRAIN_MEAN = None
_TRAIN_STD = None

def get_values(x):
    if hasattr(x,"values"):
        x = x.values
    else:
        if not isinstance(x,np.ndarray):
            x = np.array(x)
    return x