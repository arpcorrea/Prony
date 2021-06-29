# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:49:17 2021

@author: AQ66740
"""
import numpy as np
class Transf2TimeDomain:
    def __init__(self, n, optimization):  
        self.calc_time(n, optimization)
        
    # def calc_time(self, n, optimization):            
    #     t=[]
    #     for ti in range (0, 100):
    #         t.append(ti*0.00005)
    #     dict_G = {}     
    #     for data_set in optimization.dict_opt:
    #         popt = optimization.dict_opt[data_set]
    #         for i in range (1, len(popt),2):
    #             popt[i] = popt[i]/(2*np.pi)
    #         G =[]   
    #         for ti in t:   
    #             summation = 0
                             
    #             for l in range (0, n-1, 2):
    #                 # import pdb; pdb.set_trace()
    #                 summation = summation + popt[l]*np.exp(-ti/popt[l+1])
    #             G.append(summation)
    #             dict_G[data_set] = G
    #     self.dict_G = dict_G.copy()
    #     self.t = t
    
    # fig2, ax2 = plt.subplots()
    # ax2.set_xlabel('time (s)')
    # ax2.set_ylabel('Modulus [Pa]')
    # ax2.plot(t, G180, label='Prony180', color='tab:purple')
    
    # ax2.set_xlabel('time (s)')
    # ax2.set_ylabel('Modulus [Pa]')
    # ax2.plot(t, G160, label='Prony160', color='tab:green')
    
    # ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    # ax2.legend()
    
    
    def calc_time(self, n, optimization):            
        t=np.zeros(100)
        for i in range (0, len(t)):
            t[i] = i*0.005 
        dict_G = {}     
        for data_set in optimization.dict_opt:
            popt = optimization.dict_opt[data_set].copy()
            G = np.zeros((len(t),2))
            for i in range(0, len(t)):   
                summation = 0                             
                for l in range (0, n-1, 2):
                    summation = summation + popt[l]*np.exp(-t[i]/optimization.t[l])
                G[i,1] = summation
                G[i,0] = t[i]
                dict_G[data_set] = G
        self.dict_G = dict_G.copy()
        self.t = t