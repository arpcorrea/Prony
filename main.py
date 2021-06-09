# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:29:15 2021

@author: Admin
"""
from PronyModulus import *

import pandas as pd
from scipy import optimize
from scipy.optimize import curve_fit
import math
import numpy as np
import matplotlib.pyplot as plt




xl = pd.ExcelFile('data.xlsx')
data_dict={}
for name in xl.sheet_names:
    df = xl.parse(name)
    data_dict[name]=df
    
    




df = pandas.read_excel('data.xlsx', '180' )
df2 = pandas.read_excel('data.xlsx', '160' )
frequency = df.values[:,0]
storage = df.values[:,1]
loss = df.values[:,2]

# for sheet_name 

# data ={}
# for name in sheet_name:
#     data.update[name, value]


# points per decade
p=2
# number of unknown variables 
n=p*2*math.ceil(math.log10(max(frequency)-math.log10(min(frequency))))


bounds=InitialBounds(frequency,n)
        
