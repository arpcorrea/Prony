# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:19:00 2021

@author: AQ66740
"""
import numpy as np
import math
class Data:
    
    def __init__(self, xl,p):
        self.arrange_data(xl,p)
        
    def arrange_data(self, xl,p):
        data_dict={}
        
        for name in xl.sheet_names:
            df = xl.parse(name)
            for i, variable in enumerate(df):
                if  i == 0:
                    w = df[variable].tolist()
                elif i == 1:
                    storage = df[variable].tolist()
                elif i == 2:
                    loss = df[variable].tolist()
            value = np.zeros((len(w),3))
            value [:,0] = w
            value [:,1] = storage
            value [:,2] = loss
            data_dict[name] = value
            
            # number of unknown variables 
            n=p*2*math.ceil(math.log10(max(w)-math.log10(min(w))))
            
            self.data_dict = data_dict.copy()
            self.n = 6
    

