# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:38:18 2021

@author: AQ66740
"""
import numpy as np
import math

class ReptationTime:
    def __init__(self, data, opt_dict, int_func):   
        self.calc_reptation(data, opt_dict, int_func)
        
    
    def calc_reptation(self, data, opt_dict, int_func):
        w = np.array([1e-4, 1e-3, 1e-2]  ) 
        w_log = np.log10(w)
        w_reg = np.array([0.1, 1, 10, 100, 10000]) 
        
        
        
        dict_rept_time={}
        dict_reg={}
        for data_set in data.data_dict: 
            reg=np.zeros(( 5,3))
            reg[:, 0] = w_reg
            for i in w:
                str_rept = np.log10(int_func.str_func(w, opt_dict[data_set],data. n))
                ls_rept  = np.log10(int_func.loss_func(w, opt_dict[data_set],data. n))
            
            linear_coef_str = np.polyfit(w_log, str_rept, 1)
            linear_coef_ls  = np.polyfit(w_log, ls_rept, 1)    
            x = (linear_coef_ls[1] - linear_coef_str[1])/(linear_coef_str[0] - linear_coef_ls[0])
            rept_time = 1/((10**x)/(2*np.pi))
            dict_rept_time[data_set] = rept_time        
            
            for i in range (0, len(w_reg)):
                # import pdb; pdb.set_trace()
                reg[i,1] = (10**(linear_coef_str[0]*math.log10(w_reg[i]) + linear_coef_str[1]) )
                reg[i,2] = (10**(linear_coef_ls[0]*math.log10(w_reg[i]) + linear_coef_ls[1]) )
                # import pdb; pdb.set_trace()
            dict_reg[data_set] = reg
                
        self.dict_rept_time = dict_rept_time.copy()  
        self.dict_reg = dict_reg.copy()
