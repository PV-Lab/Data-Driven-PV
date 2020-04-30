# -*- coding: utf-8 -*-

#Load libraries
import gpflow
from gpflow.test_util import notebook_range
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf

# Specify font used in plots
font = 'Adobe Myungjo Std'
math_font = 'cm'

#Cutoff function
def voef(cut, slope, eff):
    if eff < cut:
        return 0
    else:
        return (eff*slope)
    
#Plotting variables
def make_plots(data, gx, gy, p_type='revenue', point=False, val=False, p_max = True):  

    sns.set(style="white", context='talk')

    
    fig = plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 28})
    plt.rcParams['xtick.major.pad'] = 12
    plt.rcParams['ytick.major.pad'] = 12
    plt.rcParams['font.family'] = font
    plt.rcParams['mathtext.fontset'] = math_font

    levels = np.linspace(np.min(data), np.max(data), 30)
    plt.contourf(gx, gy, data.reshape(gx.shape).T, cmap='viridis', levels = levels)
#

    plt.xlabel('Temperature [$^{\circ}$C]', size=24, fontname=font)
    plt.ylabel('Solvent Ratio', size=24, fontname=font)
    clb = plt.colorbar(format='%.1f')
    
    clb.ax.set_title('$\eta$ [%]', fontname=font, size=24, pad=10)
    clb.ax.tick_params(labelsize=24) 

    ind = np.unravel_index(np.argmax(data.reshape(gx.shape).T, axis=None), data.reshape(gx.shape).T.shape)    

    if point.any():
        ind = point

    best_point = np.round(np.min(data.reshape(gx.shape).T, axis=None), decimals=2)
    
    if val:
        best_point = np.round(val, decimals=1)
    
    plt.contour(gx, gy, data.reshape(gx.shape).T, levels = [ min(data), 1.10*min(data)], colors = 'white', linestyles = '--')

           
    #plt.scatter(gx[ind[0],ind[1]], gy[ind[0],ind[1]], 500, marker='*',c='r',label='Best ' + str(best_point) + '%')
    if p_max == True:
        #plt.scatter(gx[ind[0],ind[1]], gy[ind[0],ind[1]], 500, marker='*',c='r', label='Best ' + str(best_point) + '%')
        plt.scatter(point[0,0], point[0,1], 500, marker='*',c='r', label='Best ' + str(best_point[0,0]) + '%')

    plt.scatter(process_rev[:,0], process_rev[:,1], c='r', alpha=0.5, marker = 'o')
    
    legend = plt.legend(frameon=True, loc='upper left', borderpad=0.3, scatteryoffsets=[0.5], handletextpad=0.3)
    plt.setp(legend.get_texts(), color='black', size=24)
    plt.tick_params(axis="x", labelsize=24)
    plt.tick_params(axis="y", labelsize=24)
       
    
    fig.tight_layout()
    

#LOAD DATA

#JV_raw = np.loadtxt('perov_sim_nJV.txt')
#par = np.loadtxt('perov_sim_para.txt')
#par = par.T

JV_exp = np.loadtxt('perov_JV_exp.txt',delimiter=',')
JV_exp = JV_exp

v_sweep = np.linspace (0,1.2,100)
power_exp= JV_exp[:,100*2:100*3]*v_sweep
eff_exp = np.max(power_exp, axis=1)/0.98

exp_condition = pd.read_excel('prcess_label.xlsx',index_col=0)
exp_condition = exp_condition.values

#%% COMPUTE REVENUE FROM EFFICIENCY

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


for i in [70,90,110,130]:
    for j in [2,4,8]:
        idx = np.intersect1d(np.where(X_data_re[:,1]==i) ,np.where(X_data_re[:,2]==j))
        p_index.append(idx[-1])

#Positional index
p_index.insert(0,0)

#Compute revenue
rev = np.zeros((12,1))
eff_p = np.zeros((12,1))
stdv = np.zeros((12,1))
max_eff = np.zeros((12,1))
p_count = np.zeros((12,1))
p_sum = np.zeros((12,1))
p_yield = np.zeros((12,1))
cut = 0
slope = 1

#Dummy revenue calculation array
rev_calc=[]
grouping_temp=[]
groups = []
mean_calc= []
var_calc = []
unique_conditions = []

for i in range (len(p_index)-1):
    if i == 0:
        X_fit = X_data_re[p_index[i]:p_index[i+1]+1,0]
    else:
        X_fit = X_data_re[p_index[i]+1:p_index[i+1]+1,0]
    print(X_fit)
    unique_conditions.append(X_data_re[p_index[i],1:])
    
    for sample in X_fit:
        rev_calc.append(voef(cut, slope, sample))
        grouping_temp.append(voef(cut, slope, sample))
        groups.append(i)
    rev[i] = np.mean(rev_calc)
    p_yield[i] = (len(rev_calc) - rev_calc.count(0)) / len(rev_calc) #yield computation
    stdv[i] = np.std(rev_calc)
    max_eff[i] = np.max(rev_calc)
    p_count[i] = len(rev_calc)
    p_sum[i] = np.sum(rev_calc)
    
    mean_calc.append(np.mean(grouping_temp))
    var_calc.append(np.var(grouping_temp))
    grouping_temp = []
    rev_calc = []

#Create final revenue array
process_rev= []
for i in [70,90,110,130]:
    for j in [2,4,8]:
        a = [i,j]
        process_rev.append(a)
        
process_rev = np.array(process_rev)   

#Choose quantity to optimize for
#metric = rev
#metric = p_yield
metric = stdv
#metric = max_eff

process_rev = np.concatenate((process_rev, metric),axis=1)

mean_calc = np.array(mean_calc)
var_calc = np.array(var_calc)


unique_conditions = pd.DataFrame(unique_conditions, columns = ['Temp', 'Ratio'])
unique_conditions['Mean'] = mean_calc
unique_conditions['Variance'] = var_calc

rev_full = np.zeros((144,1))
for i, sample in enumerate(X_data_re[:,0]):
    rev_full[i] = voef(cut, slope, sample)

process_full = np.concatenate((X_data_re[:,1:3],rev_full),axis=1)
process_full_df = pd.DataFrame(process_full, columns = ['Temp', 'Ratio', 'Value'])   
mean_calc = np.insert(mean_calc, 0, 0, axis=0)

    
#%%

np.random.seed(1)  # for reproducibility

from sklearn.preprocessing import MinMaxScaler


scaler_ratio = MinMaxScaler()
scaler_temp = MinMaxScaler()

temp_m = np.linspace(50,140,100).reshape(-1,1)
ratio_m = np.linspace(1,9.5,100).reshape(-1,1)

temp = scaler_temp.fit_transform(temp_m).reshape(-1)
ratio =  scaler_ratio.fit_transform(ratio_m).reshape(-1)


X = np.append(scaler_temp.transform(process_rev[:,0].reshape(-1,1)), 
              scaler_ratio.transform(process_rev[:,1].reshape(-1,1)),
              axis=1)


#scaler_y = MinMaxScaler()
Y = process_rev[:,2].reshape(-1,1)

gx, gy = np.meshgrid(temp, ratio)


process_mesh= []
for i in temp:
    for j in ratio:
        b = [i,j]
        process_mesh.append(b)
        
process_mesh = np.array(process_mesh) 

gx_m, gy_m = np.meshgrid(temp_m.reshape(-1), ratio_m.reshape(-1))



#gx = process_mesh[:,0].reshape(gx_m.shape)
#gy = process_mesh[:,1].reshape(gy_m.shape)

#error = np.where(error==0, 1e-4, error) 
               
# model construction (notice that num_latent is 1)
likelihood = gpflow.likelihoods.Gaussian
kern = gpflow.kernels.Matern12(input_dim=2, variance=100000)
model = gpflow.models.GPR(X, Y, kern=kern)



opt = gpflow.train.ScipyOptimizer(method='L-BFGS-B')
opt.minimize(model)
        
mu_process, cov_process = model.predict_f(process_mesh.reshape(-1,2))

try:
    make_plots(mu_process, gx_m, gy_m, process_rev = process_rev)
    
except:
    print("Error")

mu_pred, cov_pred = model.predict_f(X)   
print("MSE is", sum((mu_pred-Y)**2))

#%% PLOT PREDICTED DISTRIBUTION

#point = scaler.transform(np.array([87,3.75]).reshape(1,2))
#
#temp = scaler_temp.fit_transform(temp_m).reshape(-1)
#ratio =  scaler_ratio.fit_transform(ratio_m).reshape(-1)
#
#scaler = MinMaxScaler()

point = np.append(scaler_temp.transform(np.array([87]).reshape(-1,1)), 
              scaler_ratio.transform(np.array([3.75]).reshape(-1,1)),
              axis=1)


val, cov_val = model.predict_f(point)
make_plots(mu_process[:,0], gx_m, gy_m, p_type='efficiency', point=np.array([92, 4.1]).reshape(1,2), val=np.array([2.04]).reshape(1,1))


plt.savefig('Fig3_var', dpi=1000)




#%%


#size = 100
#samples = model.predict_f_samples(X, size).reshape([12,size,1]).flatten()
#
#process_pred = []
#
#for i in range(process_rev.shape[0]):
#    process_pred.append(np.tile(process_rev[i,:2], (size, 1)))
#
#process_pred = pd.DataFrame(np.concatenate(process_pred, axis=0), columns = ['Temp', 'Ratio'])
#
#process_pred['Value'] = samples
#
#sns.set(style="darkgrid")
#g = sns.FacetGrid(process_pred, row="Temp", col="Ratio", margin_titles=True)
#g.map(sns.distplot, "Value", kde=True, hist=False)
#g.set(xlim=(0, 300), ylim=(0, 0.05));


#
#import GPyOpt
##Choose temperatures
#
#    
#def objective(X_eval):
#    return model.predict_f(X_eval)[0]
#
##Perform Optimization for experimental data only
#
#domain = [{'name':'X1','type':'continuous','domain':(0,0.1)},
#        {'name':'X2','type':'continuous','domain':(0,1)},
#        ]
#
#max_iter = 35
#myProblem = GPyOpt.methods.BayesianOptimization(objective, domain, 
#                                                acquisition_type = 'EI',
#                                                minimize=True,
#                                                normalize_Y = True)
#myProblem.run_optimization(max_iter)
#print('Best value located at', scaler.inverse_transform(myProblem.x_opt.reshape(1,2)))
#print('Best value is:', myProblem.fx_opt)
#
#
#
#myProblem.plot_acquisition() 

