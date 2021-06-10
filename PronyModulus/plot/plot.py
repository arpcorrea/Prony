# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:59:00 2021

@author: AQ66740
"""
import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self, data_dict, opt_dict, dict_reg, G):  
        self.plot_freq(data_dict)
        self.plot_optimize(opt_dict)
        self.plot_regression(dict_reg)
        self.plot_time(G)
    
    def plot_freq(self, data_dict):
        fig, ax1 = plt.subplots()
        ax1.set_yscale('log')
        ax1.set_xscale('log')    
        ax1.set_xlabel('frequency (rad/s)')
        ax1.set_ylabel('Modulus [Pa]')
        ax1.set_ylim([0.1, 1000000])
     
        count = 0
        for i in data_dict:
            count = count+1
        color=iter(plt.cm.rainbow(np.linspace(0,1,count)))
    
        for data_set in data_dict: 
            c=next(color)
            # import pdb; pdb.set_trace()
            ax1.scatter(data_dict[data_set][:,0], data_dict[data_set][:,1], color=c, label = 'Storage'+data_set)
            ax1.scatter(data_dict[data_set][:,0], data_dict[data_set][:,2], color=c, label = 'Loss'+data_set, marker = 's')
        ax1.legend()   
        self.ax1=ax1
        
        
    def plot_optimize(self, opt_dict):
        count = 0
        for i in opt_dict:
            count = count+1
        color=iter(plt.cm.rainbow(np.linspace(0,1,count)))
        
        for data_set in opt_dict: 
            c=next(color)
            self.ax1.plot(opt_dict[data_set][:,0], opt_dict[data_set][:,1], label='Prony'+data_set, color=c)
            self.ax1.plot(opt_dict[data_set][:,0], opt_dict[data_set][:,2], color=c)
            self.ax1.legend()  
            
     
        
    def plot_regression(self, regression):
        count = 0
        for i in regression:
            count = count+1
        color=iter(plt.cm.rainbow(np.linspace(0,1,count)))
        for data_set in regression:
            c=next(color)
            self.ax1.plot(regression[data_set][:,0], regression[data_set][:,1], 'r--', label = 'Str' + data_set + 'terminal regr', color = c )
            self.ax1.plot(regression[data_set][:,0], regression[data_set][:,2], 'r--', label = 'Ls' + data_set + 'terminal regr', color = c)
            self.ax1.legend() 
         
            
    def plot_time(self, G):
        fig, ax2 = plt.subplots()
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('Modulus [Pa]')
        ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        
        for data_set in G.keys():
            ax2.plot(G[data_set][:,0], G[data_set][:,1], label = 'Prony' + data_set)
            

        
        ax2.legend()
            
         


