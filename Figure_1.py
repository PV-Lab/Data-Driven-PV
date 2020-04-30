#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:35:10 2020

@author: isaacparker
"""

#Load libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import lognorm, gaussian_kde

# Specify font used in plots
font = 'Adobe Myungjo Std'
math_font = 'cm'

import matplotlib as mpl
mpl.style.use('seaborn-white')
mpl.rcParams['font.family'] = font
mpl.rcParams['font.size'] = 18
mpl.rcParams['mathtext.fontset'] = 'cm'


#LOAD DATA
JV_exp = np.loadtxt('perov_JV_exp.txt',delimiter=',')
JV_exp = JV_exp

v_sweep = np.linspace (0,1.2,100)
power_exp= JV_exp[:,100*2:100*3]*v_sweep
eff_exp = np.max(power_exp, axis=1)/0.98

exp_condition = pd.read_excel('prcess_label.xlsx',index_col=0)
exp_condition = exp_condition.values

#Stack data and order     
X_data = np.concatenate([eff_exp.reshape(-1,1), exp_condition],axis= 1)

p_index = []
X_data_re=[]
for i in [70,90,110,130]:
    for j in [2,4,8]:
        idx = np.intersect1d(np.where(X_data[:,1]==i) ,np.where(X_data[:,2]==j))
        X_data_re.append(X_data[idx,:])
        
X_data_re = np.vstack(X_data_re)        

#Remove data to have same # of samples:
X_data_re = np.delete(X_data_re, [0,15,21,13,14,10,12,17,12,9,7,4], 0)
X_data_re = np.insert(X_data_re, 36, [3.88, 90, 2], axis=0)
X_data_re = np.delete(X_data_re, [106,107,108,96,110,112], 0)
X_data_re = np.insert(X_data_re, 143, [5.77, 130, 8], axis=0)

# Compute efficiency and normalize
df_X1 = pd.DataFrame(X_data_re, columns=['Efficiency','Temperature','Ratio'])

df_X = df_X1.copy()

max_eff = df_X['Efficiency'].max()
    
# Normalize
df_X['Efficiency'] = df_X['Efficiency'] / max_eff

# Get mean and variance for empirical distribution
X_mean = df_X['Efficiency'].mean()
eff_data = df_X['Efficiency']

log_norm_var = eff_data.std()

# Lognormal distribution histogram
np.random.seed(6)
logn = lognorm(s=1*log_norm_var, scale = 0.7*(1-X_mean))
sample = logn.rvs (size=500)
sample[sample>1]= 1
plt.figure()
plt.hist((1-sample)*20,50,
         density=True,
         edgecolor='white',
         linewidth=1.2,
         # color='mediumseagreen',
         # alpha=0.5)
         # color=(60/255, 110/255, 135/255, 0.8))
         color='k',
         alpha=0.5)
plt.xlabel(r'Solar Cell Efficiency $\eta$ [%]', size=18, fontname=font)
plt.ylabel(r'Probability $p(\eta)$', size=18, fontname=font)
plt.xlim(left=0, right=20)
plt.yticks(np.arange(0, 0.25, step=0.1))

density = gaussian_kde((1-sample)*20)
xs = np.linspace(0,20,50)
density.covariance_factor = lambda : 1*log_norm_var
density._compute_covariance()
plt.plot(xs,density(xs),
         # color=(60/255, 110/255, 135/255))
         # color='mediumseagreen')
         color='k')
plt.tight_layout()
plt.savefig('Fig1.png',dpi=300)
plt.show()
