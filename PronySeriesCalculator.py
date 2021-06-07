# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:30:41 2021

@author: AQ66740
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:38:11 2021

@author: AQ66740
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:28:43 2021

@author: AQ66740
"""

from IPython import get_ipython
from scipy import optimize
from scipy.optimize import curve_fit
import math
import numpy as np
import matplotlib.pyplot as plt


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


loss180 = np.array([8.52E+04,
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

storage180 =np.array([5.37E+04,
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
n=p*2*math.ceil(math.log10(max(frequency)-math.log10(min(frequency))))

#Initialize lower bound and upper bound arrays
bound_lower = np.zeros(n)
bound_upper = np.ones(n)*np.infty



# populate t bounds positions 
count = 0
for i in range (1,n-1,4):
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
        
        


def fit_storage (w, G1, t1, G2, t2, G3, t3, G4, t4, G5, t5, G6, t6):
    fit_str =  (G1*w**2*t1**2)/(1+w**2*t1**2) + (G2*w**2*t2**2)/(1+w**2*t2**2) + (G3*w**2*t3**2)/(1+w**2*t3**2) + (G4*w**2*t4**2)/(1+w**2*t4**2) + (G5*w**2*t5**2)/(1+w**2*t5**2) + (G6*w**2*t6**2)/(1+w**2*t6**2) 
    return fit_str

def fit_loss (w, G1, t1, G2, t2, G3, t3, G4, t4, G5, t5, G6, t6):
    fit_loss = G1*w*t1/(1+w**2*t1**2) + G2*w*t2/(1+w**2*t2**2) + G3*w*t3/(1+w**2*t3**2) + G4*w*t4/(1+w**2*t4**2) + G5*w*t5/(1+w**2*t5**2) + G6*w*t6/(1+w**2*t6**2) 
    return fit_loss


def loss_func(w, x, n):
    summation_ls= 0
    for i in range (0,n,2):
        summation_ls = summation_ls + ((x[i]*x[i+1]*w)/(1+w**2*x[i+1]**2))
    y_ls = summation_ls
    return y_ls

def str_func(w, x, n):
    summation_str = 0
    for i in range (0,n,2): 
        summation_str = summation_str + ((x[i]*w**2*x[i+1]**2)/(1+(w**2*x[i+1]**2))) 
    y_str = summation_str
    return y_str






# popt, pcov = curve_fit(fit_loss, frequency, storage, bounds=(bound_lower, bound_upper), maxfev = 50000)
# popt, pcov = curve_fit(str_func, frequency, storage, bounds=([bound_lower], [bound_upper]), maxfev = 50000)
popt180, pcov180 = curve_fit(fit_storage, frequency, storage180, bounds=(bound_lower, bound_upper), maxfev = 350000)
popt160, pcov160 = curve_fit(fit_storage, frequency, storage160, bounds=(bound_lower, bound_upper), maxfev = 350000)



loss_fitted180 = loss_func(frequency, popt180, n)
storage_fitted180 = str_func(frequency, popt180, n)

loss_fitted160 = loss_func(frequency, popt160, n)
storage_fitted160 = str_func(frequency, popt160, n)



#PLOT

fig, ax1 = plt.subplots()

##MEASURED 
##180
ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.scatter(frequency, storage180, color='tab:red', label = 'Storage 180')
ax1.set_yscale('log')
ax1.set_xscale('log')    


ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.scatter(frequency, loss180, color = 'tab:red', label = 'Loss 180', marker = 's')
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
##180
ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.plot(frequency, loss_fitted180,  color='tab:purple')
ax1.set_yscale('log')
ax1.set_xscale('log')

ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.plot(frequency, storage_fitted180, label='Prony', color='tab:purple')
ax1.set_yscale('log')
ax1.set_xscale('log')


##160
ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.plot(frequency, loss_fitted160,  color='tab:green')
ax1.set_yscale('log')
ax1.set_xscale('log')

ax1.set_xlabel('frequency (rad/s)')
ax1.set_ylabel('Modulus [Pa]')
ax1.plot(frequency, storage_fitted160, label='Prony', color='tab:green')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend()


t=[]
for ti in range (0, 100):
    t.append(ti*0.0002)
    
G180 = []   
G160 = [] 
for ti in t:
    summation180 = 0
    summation160 = 0
    for i in range (0, n-1, 2):
        # import pdb; pdb.set_trace()
        summation180 = summation180 + popt180[i]*np.exp(-ti/popt180[i+1])
        summation160 = summation160 + popt160[i]*np.exp(-ti/popt160[i+1])
    G180.append(summation180)    
    G160.append(summation160)    

fig, ax2 = plt.subplots()
ax2.set_xlabel('time (s)')
ax2.set_ylabel('Modulus [Pa]')
ax2.plot(t, G180, label='Prony180', color='tab:purple')

ax2.set_xlabel('time (s)')
ax2.set_ylabel('Modulus [Pa]')
ax2.plot(t, G160, label='Prony160', color='tab:green')

ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax2.legend()



## Calculate reptation time   
    
w = np.array([1e-4, 1e-3, 1e-2]  ) 
w_log = np.log10(w)
for i in w:
    str180_rept = np.log10(str_func(w, popt180, n))
    ls180_rept  = np.log10(loss_func(w, popt180, n))
    str160_rept = np.log10(str_func(w, popt160, n))
    ls160_rept  = np.log10(loss_func(w, popt160, n))
    
linear_coef_str180 = np.polyfit(w_log, str180_rept, 1)
linear_coef_ls180  = np.polyfit(w_log, ls180_rept, 1)    
x = (linear_coef_ls180[1] - linear_coef_str180[1])/(linear_coef_str180[0] - linear_coef_ls180[0])
rept_180 = 1/((10**x)/(2*np.pi))


linear_coef_str160 = np.polyfit(w_log, str160_rept, 1)
linear_coef_ls160  = np.polyfit(w_log, ls160_rept, 1)    
x = (linear_coef_ls160[1] - linear_coef_str160[1])/(linear_coef_str160[0] - linear_coef_ls160[0])
rept_160 = 1/((10**x)/(2*np.pi))


reg_str180 = []
reg_loss180 =[]

reg_str160 = []
reg_loss160 =[]
w_fit = np.array([0.1, 1, 10, 100, 100000])   
for i in range (0, len(w_fit)):
   reg_str180.append (10**(linear_coef_str180[0]*math.log10(w_fit[i]) + linear_coef_str180[1]) )
   reg_loss180.append (10**(linear_coef_ls180[0]*math.log10(w_fit[i]) + linear_coef_ls180[1]) )
   reg_str160.append (10**(linear_coef_str160[0]*math.log10(w_fit[i]) + linear_coef_str160[1]) )
   reg_loss160.append (10**(linear_coef_ls160[0]*math.log10(w_fit[i]) + linear_coef_ls160[1]) )
ax1.plot(w_fit, reg_str180, 'r--', label = 'Str 180 terminal regr.', color = 'tab:red')
ax1.plot(w_fit, reg_loss180, 'r--', label = 'Ls 180 terminal regr.', color = 'tab:red')
ax1.plot(w_fit, reg_str160, 'r--', label = 'Str 180 terminal regr.', color = 'tab:blue')
ax1.plot(w_fit, reg_loss160, 'r--', label = 'Ls 180 terminal regr.', color = 'tab:blue')

ax1.set_ylim([0.1, 1000000])
leg=ax1.legend()   