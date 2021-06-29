# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:39:23 2021

@author: Admin
"""
import numpy as np
import math
class InitialBounds:
    
    def __init__(self, data):
        # import pdb; pdb.set_trace()
        self.initial_bounds(data)
        # self.initial_bounds2(data)
  
            
    def initial_bounds(self, data):
        bound_lower = np.ones(data.n)*0
        bound_upper = np.ones(data.n)*np.infty        
        bound_dict={}
        tau_dict = {}
        for data_set in data.data_dict:
            value=np.zeros((data.n,2))
            count = 0
            tau_dict[0] = 1.
            tau_dict[1] = 0.66
            tau_dict[2] = 0.33
            tau_dict[3] = 0.01
            tau_dict[4] = 0.066
            tau_dict[5] = 0.033
            for i in range (0,data.n):
                aux = 1./tau_dict[i]
                aa= find_nearest(aux, data.data_dict[data_set][:,0])
                bb=find_index(aa,data.data_dict[data_set][:,0])
                bound_upper[i] = 5.*data.data_dict[data_set][bb,1]
                bound_lower[i] = 0.2*data.data_dict[data_set][bb,1]
                count = count+1
            value[:,0] = bound_lower
            value[:,1] = bound_upper
            bound_dict[data_set] =  value   
            self.bound_dict = bound_dict.copy()      
            self.tau_dict = tau_dict.copy()
       
def find_nearest(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_index(value, array):
    array = np.asarray(array)
    for i in range(len(array)):
        if value == array[i]:
            return i