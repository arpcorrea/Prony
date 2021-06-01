# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:28:43 2021

@author: AQ66740
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

from scipy import optimize
from scipy.optimize import curve_fit
import math
import numpy as np
import matplotlib.pyplot as plt



loss = np.array([8.52E+04,
6.49E+04,
4.78E+04,
3.40E+04,
2.33E+04,
1.58E+04,
1.02E+04,
6.57E+03,
4.18E+03,
2.64E+03,
1.66E+03,
1.04E+03,
6.50E+02,
4.06E+02,
2.52E+02,
1.55E+02])

storage=np.array([5.37E+04,
3.36E+04,
1.99E+04,
1.12E+04,
5.96E+03,
3.02E+03,
1.45E+03,
6.67E+02,
2.98E+02,
1.29E+02,
5.59E+01,
2.44E+01,
1.09E+01,
4.71E+00,
2.46E+00,
1.25E+00])

frequency=np.array([300.,
188.,
118.,
73.6,
46.0,
28.8,
18.0,
11.3,
7.06,
4.42,
2.77,
1.73,
1.08,
0.678,
0.425,
0.266])


loss160 = np.array([1.25E+05,
1.03E+05,
8.20E+04,
6.25E+04,
4.59E+04,
3.28E+04,
2.23E+04,
1.49E+04,
9.72E+03,
6.24E+03,
3.97E+03,
2.51E+03,
1.58E+03,
9.90E+02,
6.18E+02,
3.85E+02
])
storage160 = np.array([1.12E+05,
7.69E+04,
5.07E+04,
3.17E+04,
1.87E+04,
1.05E+04,
5.45E+03,
2.70E+03,
1.27E+03,
5.67E+02,
2.44E+02,
1.02E+02,
4.36E+01,
2.03E+01,
7.89E+00,
4.43E+00])

# points per decade
p=2
# number of unknown variables 
n=p*2*math.ceil(math.log10(max(frequency)-math.log10(min(frequency))))+1

#Initialize lower bound and upper bound arrays
bound_lower = np.zeros(n)
bound_upper = np.zeros (n)



bound_lower [2] = 1./(2.1*max(frequency))
bound_upper [2] = 1./(2*max(frequency))
bound_lower [-1] = 1./(1.1*10**(math.ceil(math.log10(min(frequency)))))
bound_upper [-1] = 1./(10**math.ceil(math.log10(min(frequency))))
bound_upper [1] = np.infty 

# populate t bounds positions 
count = 1
# flag = 0
for i in range (4,n-1,4):
        flag = 0
        if flag == 0:
            bound_upper[i] =  1./(10**math.ceil(math.log10(max(frequency)))/(10**(count)))
            bound_lower[i] =  1/(1.1*10**math.ceil(math.log10(max(frequency)))/(10**(count)))
            flag = flag+1
        if flag == 1:
            bound_upper[i+2] =  1./((0.5*10**math.ceil(math.log10(max(frequency)))/(10**(count))))
            bound_lower[i+2] =  1/((1.1*0.5*10**math.ceil(math.log10(max(frequency)))/(10**(count))))
            flag = 0
        count = count+1
        
        
def find_index_upper(array, value):
    # import pdb; pdb.set_trace()    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if array[idx] > value:
        idx = idx+1  
    return  idx


def find_index_lower(array, value):
    # import pdb; pdb.set_trace()
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if array[idx] > value:
        idx = idx+1  
    return  idx



# populate g position
for i in range (3, len(bound_upper),2):
    value = 1./bound_upper[i+1]
    bound_upper[i] = storage[find_index_upper(frequency, value)-1]
bound_upper[0] = 1.25        
            
for i in range (1, len(bound_upper)-1,2):
    value = 1./bound_upper[i+1]
    bound_lower[i] = storage[find_index_lower(frequency, value)]

# bound_lower =[0.2   , 53700    , 1./630, 11200, 1./110 ,   5960, 1./55 ,  298, 1./11., 129 ,  1./5.5, 4.71, 1./1.1]
# bound_upper = [1.5, np.infty , 1./600, 19900, 1./100 , 11200 , 1./50 ,  667, 1./10,  298,  1./5  , 10.9, 1./1]




   
# for i in range (1, len(bound_lower)-3,2):
#     value = 1./bound_lower[i+1]
#     bound_lower[i] = storage[find_index_lower(frequency, value)+1]
    
# bound_lower[-2] = 0
# bound_lower[0] = 0



# lower_bound =[0.25       , 100000 , 0        ,  10000  , 1./300. , 1000  , 1./46   , 100  , 1./11.3    , 10   , 1/2.77  , 1  , 1./0.678   , 0.1 , 1./0.266]
# upper_bound = [np.infty, 1000000, 1./300   , 100000  , 1./46.  , 10000 , 1./11.3   , 1000 , 1./2.77  , 100  , 1/0.678 , 10 , 1./0.266 , 1   , np.infty]


def find_nearest(array, value,original_data):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if original_data[idx] > value:
        idx = idx+1
    return  idx


def fit_storage (w,G0, G1, t1, G2, t2, G3, t3, G4, t4, G5, t5, G6, t6):
    fit_str = G0 + (G1*w**2*t1**2)/(1+w**2*t1**2) + (G2*w**2*t2**2)/(1+w**2*t2**2) + (G3*w**2*t3**2)/(1+w**2*t3**2) + (G4*w**2*t4**2)/(1+w**2*t4**2) + (G5*w**2*t5**2)/(1+w**2*t5**2) + (G6*w**2*t6**2)/(1+w**2*t6**2) 
    return fit_str

def fit_loss (w,G0, G1, t1, G2, t2, G3, t3, G4, t4, G5, t5, G6, t6):
    fit_loss = G1*w*t1/(1+w**2*t1**2) + G2*w*t2/(1+w**2*t2**2) + G3*w*t3/(1+w**2*t3**2) + G4*w*t4/(1+w**2*t4**2) + G5*w*t5/(1+w**2*t5**2) + G6*w*t6/(1+w**2*t6**2) 
    return fit_loss


def loss_func(w, x, n):
    summation_ls= np.array(np.zeros(len(frequency)))
    for i in range (1,n,2):
        summation_ls = summation_ls + ((x[i]*x[i+1]*w)/(1+w**2*x[i+1]**2))
    y_ls = summation_ls
    return y_ls

def str_func(w, x, n):
    summation_str = x[0]
    for i in range (1,n,2): 
        summation_str = summation_str + ((x[i]*w**2*x[i+1]**2)/(1+(w**2*x[i+1]**2))) 
    y_str = summation_str
    return y_str




x0 = np.zeros (n)
x0[0] = storage[round(len(storage)/2)]
inital_val = math.log10(1/frequency[0])

for i in range (1,n,2):
    x0[i] = loss[round(len(loss)/2)]
    x0[i+1] = 10**(inital_val + ((math.log10(1/frequency[0])-math.log10(1/frequency[-1]))/(n+1)));
    inital_val = math.log10(x0[i+1]);
    
lower_bound = np.zeros(n)
upper_bound = np.infty*np.ones(n)

# popt, pcov = curve_fit(fit_loss, frequency, storage, bounds=(bound_lower, bound_upper), maxfev = 50000)
# popt, pcov = curve_fit(str_func, frequency, storage, bounds=([bound_lower], [bound_upper]), maxfev = 50000)
popt, pcov = curve_fit(fit_storage, frequency, storage160, bounds=(bound_lower, bound_upper), maxfev = 350000)



loss_fitted = loss_func(frequency, popt, n)
storage_fitted = str_func(frequency, popt, n)



#PLOT

fig, ax1 = plt.subplots()

##MEASURED 
##180
ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.scatter(frequency, storage, color='tab:red', label = 'Storage 180')
ax1.set_yscale('log')
ax1.set_xscale('log')    


ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.scatter(frequency, loss, color = 'tab:red', label = 'Loss 180', marker = 's')
ax1.set_yscale('log')
plt.ylim(0.01, 100000)
ax1.set_xscale('log')

##160

ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.scatter(frequency, storage160, color='tab:blue', label = 'Storage 160')
ax1.set_yscale('log')
ax1.set_xscale('log')    


ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.scatter(frequency, loss160, color = 'tab:blue', label = 'Loss 160', marker='s')
ax1.set_yscale('log')
plt.ylim(0.01, 100000)
ax1.set_xscale('log')




#FIT
ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.plot(frequency, loss_fitted, label='Loss Prony', color='tab:green')
ax1.set_yscale('log')
ax1.set_xscale('log')

ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.plot(frequency, storage_fitted, label='Storage Prony', color='tab:purple')
ax1.set_yscale('log')
ax1.set_xscale('log')


ax1.legend()

t=[]
for i in range (0,100):
    t.append(i*0.0001)
value = []
for ti in range (len(t)):
    value.append(0)
    summation = 0
    for i in range (1, len(popt)-1,2):
        summation = summation + popt[i]*np.exp(-t[ti]/popt[i+1])
    value[ti] = popt[0] + summation
        
    
fig, ax2 = plt.subplots()
ax2.plot(t, value, label='Storage Prony', color='tab:purple')
    
    


