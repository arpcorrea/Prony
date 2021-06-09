# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:39:23 2021

@author: Admin
"""
import numpy as np
import math
class InitialBounds:
    
    def __init__(self, frequency, n):
        # import pdb; pdb.set_trace()
        self.initial_bounds(frequency, n)
        
        
    def initial_bounds(self, frequency,n):
        bound_lower = np.zeros(n)
        bound_upper = np.ones(n)*np.infty  
        # populate t bounds positions 
        count = 0
        for i in range (1,n-1,4):
            flag = 0
            if flag == 0:
                bound_upper[i] =  1.5/(10**math.ceil(math.log10(max(frequency)))/(10**(count)))
                bound_lower[i] =  0.5/(10**math.ceil(math.log10(max(frequency)))/(10**(count)))
                flag = flag+1
            if flag == 1:
                bound_upper[i+2] =  0.5*1.5/((10**math.ceil(math.log10(max(frequency)))/(10**(count))))
                bound_lower[i+2] =  0.5*0.5/((10**math.ceil(math.log10(max(frequency)))/(10**(count))))
                flag = 0
            count = count+1
        
        
    def least_squred_bounds(n):
        LowerBound = np.zeros(n)
        UpperBound = np.ones(n)*np.infty