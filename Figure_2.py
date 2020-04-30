# -*- coding: utf-8 -*-

#Load libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import lognorm


# Specify font used in plots
font = 'Adobe Myungjo Std'
math_font = 'cm'

#Cutoff function
def voef(cut, slope, eff):
    if eff < cut:
        return 0
    else:
        return (eff*slope)

def voef_exp(cut, slope, eff, exp):
    if eff < cut:
        return 0
    else:
        return (slope * np.power(eff,exp))
    
def make_plots(efficiencies, share_max, axs=False):
    
   
    cuts = np.linspace(8, 13, 100)
    slopes = np.linspace(0.5, 10, 100)
      
    distributions = []
    total_rev =[]
    distributions_exp = []
    total_rev_exp =[]
    
    
    for slope in slopes:
        for cut in cuts:
        #Compute revenue
            rev = np.zeros((len(efficiencies),1))
            cut = cut
            slope = slope
        
            for i, sample in enumerate(efficiencies):
                rev[i] = voef(cut, slope, sample)
        
            distributions.append(rev)
            total_rev.append([np.sum(rev)/len(efficiencies), cut, slope])
        
    total_rev_def = pd.DataFrame(total_rev, columns = ['Total Revenue', 'n_c', 'k'])
    
    cuts_n = cuts / np.mean(cuts)
    
    X, Y = np.meshgrid(cuts_n, slopes)
    #fig,ax = plt.subplots(1,1)
    
    if axs:
        ax = axs

    z= total_rev_def['Total Revenue']
    Z= z.values.reshape(100,100)
    
    
    levels = np.linspace(0, share_max, 10)
    cp = ax.contourf(X, Y, Z, cmap='viridis', levels = levels)
    #cp.ax.set_title('$/m^2', size=20)
    clb = plt.colorbar(cp, ax=axs, format='%.1f') # Add a colorbar to a plot
    clb.ax.set_title(r'$\$ / m^2$', fontname=font) 
    #ax.set_title('Total Revenue', size=20)
    ax.set_xlabel(r'Ratio $\eta_c / \eta _{mean}$', size=18, fontname=font)
    ax.set_ylabel(r'$k$ [$\$ \: W^{-1} m^{-2}$]', size=18, fontname=font)
        
    # Set the font name for axis tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)

    return total_rev_def, X, Y, Z


#%%

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



#make_plots(X_data_re)

#%%

# Histogram of data

#plt.hist(X_data_re[:,0],50)

# Compute efficiency and normalize
df_X1 = pd.DataFrame(X_data_re, columns=['Efficiency','Temperature','Ratio'])

df_X2 = df_X1.values

df_X2 = df_X2[df_X2[:,0]>2]
df_max = np.max(df_X2[:,0])
df_X2 = df_X2[:,0] / df_max

mean_zeroff = np.mean(df_X2)
std_zeroff = np.std(df_X2)

#plt.hist(df_X2, 50)


from scipy.stats import norm

logn_zero = norm(loc=mean_zeroff, scale = std_zeroff)

sample_zero = logn_zero.rvs(size=1500)
sample_zero = sample_zero[sample_zero < 20/ df_max]

z_i = np.random.randint(10, size=len(sample_zero))
z_i[z_i>0]=1 
sample_z = z_i *(1-sample_zero)
sample_z [sample_z<0]=0

#plt.figure()
#plt.hist(sample_z * df_max,50)
#plt.axvline(x=np.mean(sample_zero)*df_max, color='r')

#plt.figure()
#plt.hist(X_data_re[:,0], 50)


#%%
#Lognormal

#logn = lognorm(s=std_zeroff, scale = (1-mean_zeroff)+ (std_zeroff**2 / 2))
#
#
#sample = logn.rvs(size=150000)

sample = np.random.lognormal(np.log(1-mean_zeroff), (std_zeroff), 1500 )
sample[sample>1]= 1
#plt.figure()
#plt.hist ((1-sample)*df_max,50)

mean = np.mean((1-sample))
#plt.axvline(x=mean*df_max, color='r')

#%%

#a = make_plots((sample_z * df_max), share_max = 80)
#b= make_plots(((1-sample)*df_max), share_max = 80)

#calc = []
#for i, sample in enumerate((1-sample)*df_max):
#     calc.append(voef(9.5, 5, sample))
#
#res = np.sum(calc)

#%%
import matplotlib.style as style


#style.use('seaborn-white')

sns.set(style="white", context='talk')
fig, axes = plt.subplots(2, 2, figsize=(10,8))


import matplotlib
matplotlib.rcParams['font.family'] = font
matplotlib.rcParams['mathtext.fontset'] = math_font
#matplotlib.rcParams['font.size'] = 20



axes[0, 0].hist(sample_z * df_max, 50, density=True, color='mediumseagreen')
axes[0, 0].axvline(x=np.mean(sample_zero)*df_max, color='blue', linewidth=2)
#axes[0, 0].set_title('R&D Distribution', size=20)
axes[0, 0].set_xlabel(r'Solar Cell Efficiency $\eta$ [%]', size=18, fontname=font)
axes[0, 0].set_ylabel(r'Probability $p\:(\eta)$', size=18, fontname=font)
axes[0, 0].set_xlim(right=20)

axes[0, 1].hist((1-sample)*df_max, 50, density=True, color='mediumseagreen')
axes[0, 1].axvline(x=mean*df_max, color='blue', linewidth=2)
#axes[0, 1].set_title('Manufacturing Distribution', size=20)
axes[0, 1].set_xlabel(r'Solar Cell Efficiency $\eta$ [%]', size=18, fontname=font)
axes[0, 1].set_ylabel(r'Probability $p\:(\eta)$', size=18, fontname=font)
axes[0, 1].set_xlim(right=20)

# Set the font name for axis tick labels
for ax in axes.ravel():
    for tick in ax.get_xticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)

make_plots((sample_z * df_max), share_max = 80, axs= axes[1,0])
make_plots(((1-sample)*df_max), share_max = 80, axs= axes[1,1])

fig.tight_layout()

plt.savefig('Fig2.png', dpi=1000)

#axes[1, 0].scatter(x, y)






#%%

#df_X = df_X1.copy()
#
#max_eff = df_X['Efficiency'].max()
#    
## Normalize
#df_X['Efficiency'] = df_X['Efficiency'] / max_eff
#
## Get mean and variance for empirical distribution
#X_mean = df_X['Efficiency'].mean()
#eff_data = df_X['Efficiency']
#
#log_norm_var = eff_data.std()
#
#make_plots(X_data_re)
#plt.figure()
#plt.hist(X_data_re[:,0], 50)
#
#
#
##%%
## Lognormal distribution
#
#
#logn = lognorm(s=0.5*log_norm_var, scale = 0.5*(1-X_mean))
#sample = logn.rvs (size=1500)
#sample[sample>1]= 1
#plt.figure()
#plt.hist (1-sample,50)
#
##%%
#
##zero inflated lognormal
#logn_zero = norm(loc=0.5*log_norm_var, scale = 1.9*(1-X_mean))
#sample_zero = logn_zero.rvs(size=1500)
#
#z_i = np.random.randint(10, size=len(sample_zero))
#z_i[z_i>0]=1 
#sample_z = z_i *(1-sample_zero)
#sample_z [sample_z<0]=0
#
#plt.figure()
#plt.hist(sample_z,50)
#
#
#
##%%
##Make data frames for plotting
#lognorm_df = 1-sample
#lognorm_df = lognorm_df.reshape(-1,1)
#lognorm_df = lognorm_df * max_eff
#a = make_plots(lognorm_df)
#
##Make data frames for plotting
#lognorm_zero = sample_z
#lognorm_zero = lognorm_zero.reshape(-1,1)
#lognorm_zero = lognorm_zero * max_eff
#b = make_plots(lognorm_zero)
#
#
#
##%%
#
#plt.figure()
#plt.hist(lognorm_df,50)
#plt.figure()
#plt.hist(lognorm_zero,50)
#
##%%

##    
#    
#    
#    sns.lineplot(x='k', y='Total Revenue', hue='n_c', data=total_rev_def, legend='full')
    
#    for exponent in exponents:
#        for cut in cuts:
#        #Compute revenue
#            rev_exp = np.zeros((len(efficiencies),1))
#            cut = cut
#            slope = fixed_slope
#            exponent = exponent
#        
#            for i, sample in enumerate(efficiencies):
#                rev_exp[i] = voef_exp(cut, slope, sample[0], exponent)
#    
#        
#            distributions_exp.append(rev_exp)
#            total_rev_exp.append([np.sum(rev_exp), cut, exponent])
#        
#    total_rev_def_exp = pd.DataFrame(total_rev_exp, columns = ['Total Revenue', 'n_c', 'Exp'])
#    
#    plt.figure()
#    sns.lineplot(x='Exp', y='Total Revenue', hue='n_c', data=total_rev_def_exp, legend='full')    

