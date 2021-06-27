# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:29:15 2021

@author: Admin
"""
from PronyModulus import *

import pandas as pd

xl = pd.ExcelFile('data.xlsx')
# number of maxwerl elements per decade
p = 2

data = Data(xl,p)

bounds=InitialBounds(data)
        
int_func = Functions()

curvefitting = CurveFit(data, bounds, int_func)

optimization = FinalOptimization(data, curvefitting.dict_opt, int_func, bounds.bound_dict)

reptation_time = ReptationTime(data, optimization.dict_opt, int_func)

transf_to_time = Transf2TimeDomain(data.n,optimization)

# plot = Plot(data.data_dict, curvefitting.dict_fitted, reptation_time.dict_reg, transf_to_time.dict_G)
plot = Plot(data.data_dict, optimization.fitted_dict, reptation_time.dict_reg, transf_to_time.dict_G)