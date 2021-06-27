# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:08:39 2021

@author: AQ66740
"""
from scipy import optimize
import numpy as np

class FinalOptimization:
    
    def __init__(self, data, x0 ,int_func, bound_dict):
        import pdb; pdb.set_trace
        self.optimization(data, x0, int_func, bound_dict)
        self.fit_points(data, int_func)
        
        
    def optimization(self, data, x0, int_func, bound_dict):   
        dict_opt = {}
        # boundsproblem = (np.zeros(data.n), np.ones(data.n)*np.infty)
        for data_set in data.data_dict:      
            # import pdb; pdb.set_trace()
            boundsproblem = (bound_dict[data_set][:,0], bound_dict[data_set][:,1])
            result = optimize.least_squares(int_func.opt_storage, x0[data_set], args = ( data.data_dict[data_set][:,0],  data.data_dict[data_set][:,1]), bounds=boundsproblem)
            params = result.x
            dict_opt[data_set] = params        
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
        self.fitted_dict = dict_fitted.copy()    