# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:16:07 2021

@author: AQ66740
"""
from scipy.optimize import curve_fit
import numpy as np

class CurveFit:
    def __init__(self, data, bounds, int_func):
        self.fitting(data,bounds,int_func)
        self.fit_points(data, int_func)
        
    def fitting (self, data, bounds, int_func):
        dict_opt = {}
        for data_set in data.data_dict:
            # popt180, pcov180 = curve_fit(fit_storage      , frequency                   , storage180                   , bounds=(bound_lower, bound_upper), maxfev = 350000)
            popt,       pcov = curve_fit(int_func.fit_storage, data.data_dict[data_set][:,0], data.data_dict[data_set][:,1], bounds=(bounds.bound_dict[data_set][:,0], bounds.bound_dict[data_set][:,1]), maxfev = 350000)
            dict_opt[data_set] = popt        
        self.dict_opt = dict_opt.copy()
        
    def fit_points(self, data, int_func):
        dict_fitted = {}
        for data_set in data.data_dict:
            # import pdb; pdb.set_trace()
            value = np.zeros((len(data.data_dict[data_set][:,0]),3))
            value[:,0] = data.data_dict[data_set][:,0]
            value[:,1] = int_func.str_func(value[:,0], self.dict_opt[data_set], data.n)
            value[:,2] = int_func.loss_func(value[:,0], self.dict_opt[data_set], data.n)
            dict_fitted[data_set] = value
        self.dict_fitted = dict_fitted.copy()
        
 