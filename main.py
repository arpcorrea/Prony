# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:29:15 2021

@author: Admin
"""
from PronyModulus import *

import pandas
from scipy import optimize
from scipy.optimize import curve_fit
import math
import numpy as np
import matplotlib.pyplot as plt

df=pandas.read_excel('data180.xlsx')
frequency = df.values[:,0]
storage = df.values[:,1]
loss = df.values[:,2]


# points per decade
p=2
# number of unknown variables 
n=p*2*math.ceil(math.log10(max(frequency)-math.log10(min(frequency))))


bounds=InitialBounds(frequency,n)
        
