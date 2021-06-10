# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:07:46 2021

@author: AQ66740
"""

class Functions:
    def fit_storage (self, w, G1, t1, G2, t2, G3, t3, G4, t4, G5, t5, G6, t6):
        fit_str =  (G1*w**2*t1**2)/(1+w**2*t1**2) + (G2*w**2*t2**2)/(1+w**2*t2**2) + (G3*w**2*t3**2)/(1+w**2*t3**2) + (G4*w**2*t4**2)/(1+w**2*t4**2) + (G5*w**2*t5**2)/(1+w**2*t5**2) + (G6*w**2*t6**2)/(1+w**2*t6**2) 
        return fit_str

    def fit_loss (w, G1, t1, G2, t2, G3, t3, G4, t4, G5, t5, G6, t6):
        fit_loss = G1*w*t1/(1+w**2*t1**2) + G2*w*t2/(1+w**2*t2**2) + G3*w*t3/(1+w**2*t3**2) + G4*w*t4/(1+w**2*t4**2) + G5*w*t5/(1+w**2*t5**2) + G6*w*t6/(1+w**2*t6**2) 
        return fit_loss
    
    
    def loss_func(self, w, x, n):
        summation_ls= 0
        for i in range (0,n,2):
            summation_ls = summation_ls + ((x[i]*x[i+1]*w)/(1+w**2*x[i+1]**2))
        y_ls = summation_ls
        return y_ls
    
    def str_func(self, w, x, n):
        summation_str = 0
        for i in range (0,n,2): 
            summation_str = summation_str + ((x[i]*w**2*x[i+1]**2)/(1+(w**2*x[i+1]**2))) 
        y_str = summation_str
        return y_str
    
    
    
    def opt_storage (self, x, w, s):
        opt_str =  (x[0]*w**2*x[1]**2)/(1+w**2*x[1]**2) + (x[2]*w**2*x[3]**2)/(1+w**2*x[3]**2) + (x[4]*w**2*x[5]**2)/(1+w**2*x[5]**2) + (x[6]*w**2*x[7]**2)/(1+w**2*x[7]**2) + (x[8]*w**2*x[9]**2)/(1+w**2*x[9]**2) + (x[10]*w**2*x[11]**2)/(1+w**2*x[11]**2) 
        return opt_str - s

    def opt_loss (self, x, w, l):
        opt_loss = x[0]*w*x[1]/(1+w**2*x[1]**2) + x[2]*w*x[3]/(1+w**2*x[3]**2) + x[4]*w*x[5]/(1+w**2*x[5]**2) + x[6]*w*x[7]/(1+w**2*x[7]**2) + x[8]*w*x[9]/(1+w**2*x[9]**2) + x[10]*w*x[11]/(1+w**2*x[11]**2) 
        return opt_loss - l