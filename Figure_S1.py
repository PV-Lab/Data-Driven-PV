# -*- coding: utf-8 -*-

#Load libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import interpolate as interp
from qbstyles import mpl_style

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
def make_plots(data, gx, gy, p_type='revenue', p_max = True):  

    #mu_process[mu_process<0] = 0
    
    plt.figure()
    plt.style.use(['seaborn'])
    plt.rcParams["figure.figsize"] = [4, 3]
    plt.rcParams.update({'font.size': 28})
    plt.rcParams['xtick.major.pad'] = 8
    plt.rcParams['ytick.major.pad'] = 8
    plt.rcParams['font.family'] = font
    plt.rcParams['mathtext.fontset'] = math_font

    levels = np.linspace(np.min(data), np.max(data), 30)
    plt.contourf(gx, gy, data.reshape(gx.shape), cmap='viridis', levels = levels)
#

    plt.xlabel(r'Temp [$^\circ$C]', size=20)
    plt.ylabel('Solvent Ratio', size=20)
    clb = plt.colorbar()
    
    clb.ax.tick_params(labelsize=15) 

    if p_type == 'efficiency':
        clb.ax.set_title(r'$\\eta$[%]', size=20)
    elif p_type == 'revenue':
        clb.ax.set_title(r'R[$/m^2]', size=20)        

    ind = np.unravel_index(np.argmax(data.reshape(gx.shape), axis=None), data.reshape(gx.shape).shape)

    check = data.reshape(gx.shape)
    np.max(check)

    best_point = np.round(np.max(data.reshape(gx.shape), axis=None), decimals=2)

    #plt.scatter(gx[ind[0],ind[1]], gy[ind[0],ind[1]], 500, marker='*',c='r',label='Best ' + str(best_point) + '%')
    if p_max == True:
        plt.scatter(gx[ind[0],ind[1]], gy[ind[0],ind[1]], 500, marker='*',c='r',label='Best  ' + str(best_point))

    legend = plt.legend(frameon=True)
    plt.setp(legend.get_texts(), color='black', size=20)
    plt.tick_params(axis="x", labelsize=15)
    plt.tick_params(axis="y", labelsize=15)

    #plt.show()

#% LOAD DATA

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
    unique_conditions.append(X_data_re[p_index[i]+1,1:])
    
    for sample in X_fit:
        rev_calc.append(voef(cut, slope, sample))
        grouping_temp.append(voef(cut, slope, sample))
        groups.append(i)
    rev[i] = np.mean(rev_calc) 
    #rev[i+1] = (len(rev_calc) - rev_calc.count(0)) / len(rev_calc) #yield computation
    stdv[i] = np.std(rev_calc)
    max_eff[i] = np.max(rev_calc)
    p_count[i] = len(rev_calc)
    
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
process_rev = np.concatenate((process_rev,rev),axis=1)

mean_calc = np.array(mean_calc)
var_calc = np.array(var_calc)


unique_conditions = pd.DataFrame(unique_conditions, columns = ['Temp', 'Ratio'])
unique_conditions['Mean'] = mean_calc
unique_conditions['Variance'] = var_calc

rev_full = np.zeros((144,1))
for i, sample in enumerate(X_data_re[:,0]):
    rev_full[i] = voef(cut, slope, sample)

process_full = np.concatenate((X_data_re[:,1:3],rev_full),axis=1)
process_full_df = pd.DataFrame(process_full, columns = ['Temp.', 'Solvent Ratio', 'Efficiency (%)'])   
mean_calc = np.insert(mean_calc, 0, 0, axis=0)

#%% PLOT DISTRIBUTION OF REVENUE
import matplotlib as mpl
sns.set(style="darkgrid", context='talk')
sns.set_palette("deep")
sns.set(font=font, rc=mpl.rc('mathtext',fontset=math_font))
bins = np.linspace(0,20,30)
g = sns.FacetGrid(process_full_df, row="Temp.", col="Solvent Ratio", 
                  margin_titles=True, sharey=False, sharex=False,
                  aspect=1.5)
g.map(sns.distplot, 'Efficiency (%)', bins=bins, kde=False, rug=True)

for ax, mean in zip(g.axes.flat, process_rev[:,2]):
    ax.axvline(x=mean)
#for i, ax in enumerate(g.axes.flat):
#    plt.text(1, 1, 'Mean_Eff ' + str(np.around(rev[i], decimals=2)), horizontalalignment='right', 
#         verticalalignment='top', transform=ax.transAxes, color='forestgreen')
#    plt.text(1, 0.85, 'Std_Eff ' + str(np.around(stdv[i], decimals=2)), horizontalalignment='right', 
#         verticalalignment='top', transform=ax.transAxes, color='royalblue')
#    plt.text(1, 0.70, 'Max_Eff ' + str(np.around(max_eff[i], decimals=2)), horizontalalignment='right', 
#         verticalalignment='top', transform=ax.transAxes, color='indianred')
#    plt.text(1, 0.55, '#_of_Samples ' + str(p_count[i]), horizontalalignment='right', 
#         verticalalignment='top', transform=ax.transAxes, color='darkorange')
g.savefig('FigS2.png', dpi=1000)


stats_table = pd.DataFrame()  
stats_table['Solvent Ratio'] = process_rev[:,1]
stats_table['Annealing Temperature (C)'] = process_rev[:,0]
stats_table['Maximum Efficiency (%)'] = max_eff

efficiency_res = pd.DataFrame(process_rev)
efficiency_res['Mean Efficiency (%)'] = rev
efficiency_res['Std. Deviation Efficiency (%)'] = stdv
efficiency_res['Max. Efficiency (%)'] = max_eff
efficiency_res['Efficiency STD-Error'] = stdv / np.sqrt(12)




    
#%% Recompute with revenue and cutoff

#Stack data and order     
#X_data = np.concatenate([eff_exp.reshape(-1,1), exp_condition],axis= 1)
#
#X_data_re = np.vstack((X_data[124:,:],X_data[112:124,:],X_data[101:112,:],X_data[89:101,:],X_data[77:89,:]
#, X_data[65:77,:],X_data[53:59,:],X_data[:6,:],X_data[6:18,:],X_data[59:65,:],X_data[18:53,:]))
#
##Positional index
#p_index = [0,10,22,33,45,57,69,81,99,111,123,134]

#Compute revenue
rev = np.zeros((12,1))
eff_p = np.zeros((12,1))
stdv = np.zeros((12,1))
max_eff = np.zeros((12,1))
p_count = np.zeros((12,1))
p_sum = np.zeros((12,1))
p_yield = np.zeros((12,1))
cut = 12
slope = 3

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
process_rev = np.concatenate((process_rev,rev),axis=1)

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
#Plot distributions of revenue with statistics
mpl_style(dark=False)
sns.set(style="darkgrid", context='talk')
sns.set_palette("deep")
sns.set(font=font, rc=mpl.rc('mathtext',fontset=math_font))
bins = np.linspace(0,70,30)
g = sns.FacetGrid(process_full_df, row="Temp", col="Ratio", margin_titles=True, sharey=False)
g.map(sns.distplot, "Value", bins=bins, kde=False, rug=True)
for ax, mean in zip(g.axes.flat, process_rev[:,2]):
    ax.axvline(x=mean)
#for i, ax in enumerate(g.axes.flat):
#    plt.text(1, 1, 'Mean_Rev ' + str(np.around(rev[i], decimals=2)), horizontalalignment='right', 
#         verticalalignment='top', transform=ax.transAxes, color='forestgreen')
#    plt.text(1, 0.90, 'Std_Rev ' + str(np.around(stdv[i], decimals=2)), horizontalalignment='right', 
#         verticalalignment='top', transform=ax.transAxes, color='royalblue')
#    plt.text(1, 0.80, 'Max_Rev ' + str(np.around(max_eff[i], decimals=2)), horizontalalignment='right', 
#         verticalalignment='top', transform=ax.transAxes, color='indianred')
#    plt.text(1, 0.70, '#_of_Samples ' + str(p_count[i]), horizontalalignment='right', 
#         verticalalignment='top', transform=ax.transAxes, color='darkorange')
#    plt.text(1, 0.60, 'Total_Rev ' + str(np.around(p_sum[i], decimals=2)), horizontalalignment='right', 
#         verticalalignment='top', transform=ax.transAxes, color='darkred')
#    plt.text(1, 0.50, 'Yield ' + str(np.around(p_yield[i], decimals=2)), horizontalalignment='right', 
#         verticalalignment='top', transform=ax.transAxes, color='salmon')
 
    
    #%%
    
    
#efficiency_res = pd.DataFrame(process_rev)
efficiency_res['Mean Revenue ($/%m2)'] = rev
efficiency_res['Std. Deviation Rev'] = stdv
efficiency_res['Max. Revenue (%)'] = max_eff
efficiency_res['No. of Samples'] = p_count
efficiency_res['Total Revenue'] = p_sum
efficiency_res['Yield'] = p_yield
efficiency_res['Total Revenue STD-Error'] = stdv * np.sqrt(12)

efficiency_res.to_csv('Table-1.csv')



