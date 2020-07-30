# -*- coding: utf-8 -*-


from keras import backend as K
from keras.models import Sequential


from keras.models import Model
from keras import metrics
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from keras.layers import Input, Dense, Lambda,Conv1D,Conv2DTranspose, LeakyReLU,GlobalAveragePooling1D,Activation,Flatten,Reshape
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

import seaborn as sns

import pandas as pd

from scipy import interpolate as interp
import os
from keras import optimizers
os.environ["MKL_THREADING_LAYER"] = "GNU"

K.clear_session()


JV_raw = np.loadtxt('perov_sim_nJV.txt')
JV_raw =JV_raw [:,200:-200]

par = np.loadtxt('perov_sim_para.txt')
par = par.T

'''
constrained AE
'''

def Conv1DTranspose(input_tensor, filters, kernel_size, strides ):
    
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),padding='SAME')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x



from sklearn.preprocessing import MinMaxScaler

def min_max(x):
    min = np.min(x,axis=1)
    max = np.max(x,axis=1)
    
    return (x)/max[:,None],max,min
#
JV_norm,JV_max,JV_min = min_max(JV_raw)

def log_JV(x):
    return np.log10(x+1)
JV_log = log_JV(JV_norm)

plt.figure()
#plt.plot(JV_log[100,:])
#plt.plot(JV_norm[100,:])
scaler_par = MinMaxScaler()
par_n = scaler_par.fit_transform(par) 
#par_exp_unnorm = np.loadtxt('GaAs_exp_label.txt')
#par_exp_unnorm = par_exp_unnorm.T
#
##epsilon_std = 1
#par_exp_ln = log10_ln(par_exp_unnorm)
#
#par_exp = scaler.transform(log10_ln(par_exp_unnorm ))   

    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(JV_norm,par_n, test_size=0.1)

K.clear_session()
input_dim = X_train.shape[1]
label_dim = y_train.shape[1]

x = Input(shape=(input_dim,))


y = Input(shape =(label_dim,))

max_filter = 256

strides = [5,2,2]
kernel = [7,5,3]


def encoder(x):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)    
    en0 = Conv1D(max_filter//4,kernel[0],strides= strides[0], padding='SAME')(x)
    en0 = LeakyReLU(0.2)(en0)
    en1 = Conv1D(max_filter//2,kernel[1],strides=strides[1], padding='SAME')(en0)
    en1 = LeakyReLU(0.2)(en1)
    en2 = Conv1D(max_filter,kernel[2], strides=strides[2],padding='SAME')(en1)
    en2 = LeakyReLU(0.2)(en2)
    en3 = Flatten()(en2)
    en3 = Dense(100,activation = 'relu')(en3)
    z_mean = Dense(label_dim,activation = 'linear')(en3)
   
    
    return z_mean


    


#def sampling(args):
#    z_mean, z_log_var =args
#    epsilon = K.random_normal(shape = (K.shape(z_mean)[0],label_dim),mean=0., stddev = epsilon_std)
#    return z_mean+K.exp(0.5*z_log_var/2)*epsilon

z_mean = encoder(x)
encoder_ = Model(x,z_mean)
encoder_.summary()

map_size = K.int_shape(encoder_.layers[-4].output)[1]
#z = Lambda(sampling, output_shape=(label_dim,))([z_mean,z_log_var])



# do this for recalling the decoder later

z_in = Input(shape=(label_dim,))
z1 = Dense(100,activation = 'relu')(z_in)


z1 = Dense(max_filter*map_size,activation='relu')(z1)
z1 = Reshape((map_size,1,max_filter))(z1)

#x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
#    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),padding='SAME')(x)

z2 =  Conv2DTranspose( max_filter//2, (kernel[2],1), strides=(strides[2],1),padding='SAME')(z1)
z2 = Activation('relu')(z2)

z3 = Conv2DTranspose(max_filter//4, (kernel[1],1), strides=(strides[1],1),padding='SAME')(z2)
z3 = Activation('relu')(z3)

z4 = Conv2DTranspose(1, (kernel[0],1), strides=(strides[0],1),padding='SAME')(z3)
decoded_x = Activation('sigmoid')(z4)

decoded_x = Lambda(lambda x: K.squeeze(x, axis=2))(decoded_x)
decoded_x = Lambda(lambda x: K.squeeze(x, axis=2))(decoded_x)



    
decoder_ = Model(z_in,decoded_x)
decoder_.summary()

decoded_x = decoder_(y)

ae = Model(inputs= [x,y],outputs= [decoded_x,z_mean])





def ae_loss(x, decoded_x): 
#encoder loss

    encoder_loss = K.sum(K.square(z_mean-y),axis=-1)/label_dim
    
    #decoder loss
    decoder_loss = 0.01*K.sum(K.square(x- decoded_x),axis=-1)/input_dim
    
    
    #
    ##KL loss
    #kl_loss = K.mean(-0.5* K.sum(1+z_log_var-K.square(z_mean-y)-K.exp(z_log_var),axis=-1))
    
    
    ae_loss = K.mean(encoder_loss+decoder_loss)
    
    return ae_loss
#ae.add_loss(ae_loss)

ae.compile(optimizer = optimizers.rmsprop(lr=1e-3), loss= ae_loss)
ae.summary()
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                                      patience=5, min_lr=0.00001)
ae.fit(([X_train,y_train]),([X_train,y_train]),shuffle=True, 
        batch_size=32,epochs = 500,
        validation_split=0.0, validation_data=None, callbacks=[reduce_lr])



#build encoder


encoder_ = Model(x,z_mean)

x_test_encoded_1 = encoder_.predict(X_test)


plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded_1[:, 0], y_test[:, 0])

plt.show()


x_test_decoded_1 = decoder_.predict(y_test)

x_test_decoded, x_test_encoded = ae.predict([X_test,y_test])

plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 1], x_test_encoded_1[:, 1])

plt.show()
#
#plt.plot(x_test_decoded[100,:])
##plt.plot(x_test_decoded_1[10,:])
#plt.plot(X_test[100,:])
#plt.show()


y_hat_train = decoder_.predict(y_train)
y_hat_test = decoder_.predict(y_test)
# voltage sweep unified





v_sweep = np.linspace (0,1.2,100)

v_total =np.tile(v_sweep,4).reshape(1,-1)

p_total = np.multiply(JV_norm,v_total)

sim_eff = np.max(p_total,axis=1)
np.save('perov_sim_eff',sim_eff)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error
mse_train = mse(y_hat_train,X_train)
mse_test = mse(y_hat_test,X_test)

print ('train mse: %.6f' % (mse_train))
print ('test mse: %.6f' % (mse_test))

ae.save('perov_AE.h5')
encoder_.save('perov_en.h5')

decoder_.save('perov_De.h5')
#%%
from keras.models import load_model
decoder_ = load_model('perov_De.h5')
encoder_  = load_model('perov_en.h5')


#%%

i_par= np.random.randint(100)
plt.figure()

plt.plot(X_test[i_par,:])
plt.plot(y_hat_test[i_par,:],'--')

JV_exp = np.loadtxt('perov_JV_exp.txt',delimiter=',')
JV_exp = JV_exp[:,200:-200]


v_sweep = np.linspace (0,1.2,100)
aaa= JV_exp[:,100*0:100*1]

power_exp= aaa*v_sweep
eff_exp = np.max(power_exp, axis=1)/0.98

idx_e = eff_exp>5

JV_exp = JV_exp[idx_e,:]

#plt.hist(eff_exp,20)
#%%
JV_exp_n,JV_exp_max,JV_exp_min = min_max(JV_exp)
#JV_exp_n = log_JV(JV_exp_n)

en_exp = encoder_.predict(JV_exp_n)


JV_rec = decoder_.predict(en_exp)


#get materail parameters from surrogate model
index = []
en_new = []
for ind,row in enumerate(en_exp ):
    if all(-0.2<row) and all (row<1.2):
        en_new.append(row)
        index.append(ind)
        
en_new = np.vstack(en_new)





#%%


#plt.plot(JV_exp[10,:])

    
exp_condition = pd.read_excel('prcess_label.xlsx',index_col=0)

exp_condition = exp_condition.values

exp_condition_f = exp_condition[idx_e,:][index,:]
#exp_condition_f = exp_condition

para_f = np.concatenate((exp_condition_f,en_new),axis=1)

para_mean = []
for i in [70,90,110,130]:
    for j in [2,4,8]:
        idx1 = np.intersect1d(np.where(para_f[:,0]==i) ,np.where(para_f[:,1]==j))
        para_mean.append(np.mean(para_f[idx1,:],axis=0))
        
para_mean = np.vstack(para_mean)

#exp_t= [-1,5,17,29,41,52,58,64,76,88,100,111,123,134] 

#Cutoff function
def voef(cut, slope, eff):
    if eff < cut:
        return 0
    else:
        return (eff*slope)



#Stack data and order     
X_data = np.concatenate([eff_exp.reshape(-1,1), exp_condition],axis= 1)
p_index = []
X_data_re=[]
for i in [70,90,110,130]:
    for j in [2,4,8]:
        idx = np.intersect1d(np.where(X_data[:,1]==i) ,np.where(X_data[:,2]==j))
        X_data_re.append(X_data[idx,:])
        
X_data_re = np.vstack(X_data_re)        
#Remove data or add data to have same # of samples:
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
cut = 12 #cut off value c
slope = 3 # value k in graph
#Dummy revenue calculation array
rev_calc=[]
grouping_temp=[]
groups = []
mean_calc= []
var_calc = []
unique_conditions = []


for i in range (len(p_index)-1):
    eff = []
    if i == 0:
        X_fit = X_data_re[p_index[i]:p_index[i+1]+1,0]
    else:
        X_fit = X_data_re[p_index[i]+1:p_index[i+1]+1,0]
#    print(X_fit)
    unique_conditions.append(X_data_re[p_index[i],1:])
    
    for sample in X_fit:
        rev_calc.append(voef(cut, slope, sample))
        grouping_temp.append(voef(cut, slope, sample))
        eff.append(voef(0, 1, sample))
        groups.append(i)
    rev[i] = np.mean(rev_calc)
    p_yield[i] = (len(rev_calc) - rev_calc.count(0)) / len(rev_calc) #yield computation
    stdv[i] = np.std(rev_calc)
    max_eff[i] = np.max(eff)
    p_count[i] = len(rev_calc)
    p_sum[i] = np.sum(rev_calc)
    
    mean_calc.append(np.mean(grouping_temp))
    var_calc.append(np.var(grouping_temp))
    grouping_temp = []
    rev_calc = []
#Create final revenue array
  
 
#Choose quantity to optimize for
#metric = rev
#metric = p_yield
#metric = stdv
#metric = max_eff
#metric = p_sum
#%%
metric  = [rev,max_eff]
best_p = []
mse = []
#%%

import gpflow
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

#get best porcess conditon for all metrics
for val in metric:
    process_rev= []
    for i in [70,90,110,130]:
        for j in [2,4,8]:
            a = [i,j]
            process_rev.append(a)
    process_rev = np.array(process_rev)  
    process_rev = np.concatenate((process_rev, val),axis=1)
    mean_calc = np.array(mean_calc)
    var_calc = np.array(var_calc)
#    unique_conditions = pd.DataFrame(unique_conditions, columns = ['Temp', 'Ratio'])
#    unique_conditions['Mean'] = mean_calc
#    unique_conditions['Variance'] = var_calc
#    rev_full = np.zeros((144,1))
#    for i, sample in enumerate(X_data_re[:,0]):
#        rev_full[i] = voef(cut, slope, sample)
#    process_full = np.concatenate((X_data_re[:,1:3],rev_full),axis=1)
#    process_full_df = pd.DataFrame(process_full, columns = ['Temp', 'Ratio', 'Value'])   
#    mean_calc = np.insert(mean_calc, 0, 0, axis=0)

    np.random.seed(1)  # for reproducibility
    
    temp = np.linspace(60,140,100)
    ratio = np.linspace(1,9,100)
    lengths = np.logspace(0.001, 7, num=30)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(process_rev[:,:2])
    scaler_y = MinMaxScaler()
    Y = process_rev[:,2]
    scaler_2 = MinMaxScaler()
    temp = scaler_2.fit_transform(temp.reshape(-1,1)).reshape(100,)
    ratio = scaler_2.fit_transform(ratio.reshape(-1,1)).reshape(100,)
    gx, gy = np.meshgrid(temp, ratio)
     
    temp_m = np.linspace(60,140,100)
    ratio_m = np.linspace(1,9,100) 
    gx_m, gy_m = np.meshgrid(temp_m, ratio_m)
    process_mesh= []
    for i in temp:
        for j in ratio:
            b = [i,j]
            process_mesh.append(b)
            
    process_mesh = np.array(process_mesh) 
    gx= process_mesh[:,0].reshape(gx.shape)
    gy = process_mesh[:,1].reshape(gx.shape)

    #error = np.where(error==0, 1e-4, error)          
    # model construction (notice that num_latent is 1)
#    likelihood = gpflow.likelihoods.Gaussian
#    kern = gpflow.kernels.Matern32(input_dim=2, variance=100000)
#    model = gpflow.models.GPR(X, Y, kern=kern)
#    opt = gpflow.train.ScipyOptimizer(method='L-BFGS-B')
#    opt.minimize(model)
    # build the natural gradients optimiser
    # let's do some plotting!
#    mu, var = model.predict_f(process_mesh)
#    kr = KernelRidge(kernel='rbf',alpha= 10, gamma =10000)

    
   
      
    

    svr = GridSearchCV(SVR(kernel='rbf', gamma=1), cv=3,
                   param_grid={"C": np.logspace(-4, 4, 30),
                               "gamma": np.logspace(-4, 4, 30)})
    svr.fit(X,Y)
    mu = svr.predict(process_mesh)
    best_point = process_mesh[np.argmax(mu),:]
    best_p.append(best_point)
    
    mu_predict = svr.predict(X)
#    mu_predict, cov_predict = model.predict_f(X)   
    print("MSE is", sum(( mu_predict-Y)**2))

#%%    

best_p = np.vstack(best_p)
#get fitted material parameters
Y_para = np.power(10,(scaler_par.inverse_transform(para_f[:,2:])))

#Y_para[:,0] = Y_para[:,0]*1e9

df_para = np.concatenate((para_f[:,:2],Y_para), axis=1)
df_para = pd.DataFrame(df_para, columns=['Temp', 'Sol','tau','FSRV','RSRV','Rs','Rsh' ])

key = ['Temp','Sol']

group_mean = df_para.groupby(key).mean()

group_mean.to_csv('group_mean.csv')
group_std = df_para.groupby(key).std()
group_std.to_csv('group_std.csv')
#perform GP fitting on materail parameters against process variables

X_para = scaler.transform(para_f[:,:2])

#%%
K.clear_session()
kern_par = gpflow.kernels.Matern52(input_dim=2, variance=1)
model_par = gpflow.models.GPR(X_para, para_f[:,2:], kern=kern_par)
opt_par = gpflow.train.ScipyOptimizer(method='L-BFGS-B')
opt_par.minimize(model_par)

mu_par, cov_par = model_par.predict_f(X_para)   
print("MSE is", sum((mu_par-para_f[:,2:])**2))

par_best,var_best = model_par.predict_f(best_p)


par_best_n = scaler_par.inverse_transform(par_best)

df_best_para = np.concatenate((scaler.inverse_transform(best_p),np.power(10,par_best_n)), axis=1)
df_best_para = pd.DataFrame(df_best_para, columns=['Temp', 'Sol','tau','FSRV','RSRV','Rs','Rsh' ])

df_best_para.to_csv('best_parameters.csv')
#% PLOTTING

#pentagon
from math import pi

plt.rcParams["figure.figsize"] = [12, 8]
fig = plt.figure()

# Set data
df = pd.DataFrame(par_best, columns= ['tau','FSRV','RSRV','Rs','Rsh'])

df['group'] = ['rev','max_eff']
 
df = df [['group', 'tau','FSRV','RSRV','Rs','Rsh']]
 
# ------- PART 1: Create background
 
# number of variable
categories=[ r'$\tau$',r'$^{FSRV}$',r'$^{RSRV}$',r'$^{R_{s}}$',r'$^{R_{sh}}$']
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories,size=40)
 
# Draw ylabels
ax.set_rlabel_position(0)

plt.yticks([0.3,0.5,0.7], ["0.3","0.5","0.7"], color="grey", size=30)
plt.ylim(0,1.1)
 
 
# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Total Maximum Revenue")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Maximum Efficiency")
ax.fill(angles, values, 'r', alpha=0.1)

plt.rc('legend',**{'fontsize':26})
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5) 
# Add legend
# plt.legend(loc='upper right')
fig.tight_layout()
fig.savefig('figure5.png', format='png',dpi=600)  


#%%




#contourf

levels = np.linspace(np.min(mu), np.max(mu), 30)
plt.contourf(gx_m, gy_m, mu.reshape(gx.shape), cmap='viridis', levels = levels)
#

plt.xlabel('Temperature [$^\circ$C]', size=20)
plt.ylabel('Solvent Ratio', size=20)








#plot_data = np.concatenate([eff_exp.reshape(-1,1), exp_condition],axis= 1)
#
#plot_data_re = np.vstack((plot_data[124:,:],plot_data[112:124,:],plot_data[101:112,:],plot_data[89:101,:],plot_data[77:89,:]
#, plot_data[65:77,:],plot_data[53:59,:],plot_data[:6,:],plot_data[6:18,:],plot_data[59:65,:],plot_data[18:53,:]))
#
#p_index = [0,10,22,33,45,57,69,81,99,111,123,134]
#
#
##first GP model to model the eff distribution
#
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import ConstantKernel, Matern
#
#from sklearn.neighbors import KernelDensity
#
#
#
#
#
#
##%%
#
## Compute posterior predictive mean and covariance
#
#
#def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
#    X = X.ravel()
#    mu = mu.ravel()
#    uncertainty = 1.96 * np.sqrt(np.diag(cov))
#    
#    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
#    plt.plot(X, mu, label='Mean')
#def plot_gp_2D(gx, gy, mu, X_train, Y_train):
#    
#    plt.contourf(gx, gy, mu.reshape(gx.shape))
#    plt.scatter(X_train[:,0], X_train[:,1], Y_train)#, c=Y_train)
##    plt.colorbar()
##    ax.set_title(title)
#
## Plot the results
##cut_t = [15,10,0]  
##
##slope = [1,0.4,]  
#rev = np.zeros((12,1))
#eff_p = np.zeros((12,1))
#cut=0
#slope = 1
#
##%%
#for i in range (len(p_index)-1):
#    X_fit = plot_data_re[p_index[i]:p_index[i+1]+1,0]
#    freq,eff_x= np.histogram(X_fit,bins=5)
#    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X_fit.reshape(-1,1))
#   
#    
# 
##    gpr.fit(eff_x[1:].reshape(-1,1), freq.reshape(-1,1))
#    eff_train = np.linspace( eff_x[0], eff_x[-1],100).reshape(-1,1)
#    dens = np.exp(kde.score_samples(eff_train ))
#    dens = dens/sum(dens)
#    
##    mu_s, cov_s = gpr.predict(eff_train, return_cov=True)
#    
##    plt.figure(i)
##    plt.plot(eff_train,dens)
##    plot_gp(mu_s/sum(mu_s),cov_s,eff_train)
##    plt.scatter(eff_x[:-1], freq/sum(freq))
##    mu_s[mu_s<0]=0
#    
#    rev[i+1] = sum(dens.reshape(-1,1)*voef(cut,slope, eff_train))
#    eff_p[i+1] = max(X_fit)
#max_idx = np.argmax(rev)
#print(max_idx)
#
#process_rev= []
#for i in [70,90,110,130]:
#    for j in [2,4,8]:
#        a = [i,j]
#        process_rev.append(a)
#        
#process_rev = np.array(process_rev)   
###only used when max eff
#rev = eff_p     
#
#process_rev = np.concatenate((process_rev,rev),axis=1)
#
##second GP to predict the revenue vs process
#m52 = ConstantKernel(1) * Matern(length_scale=1, nu=3)
#gpr_process = GaussianProcessRegressor(kernel=m52, alpha=0.01)
#
#
#gpr_process.fit(process_rev[:,:2], process_rev[:,2].reshape(-1,1))
#
#temp = np.linspace(60,140,100)
#
#ratio = np.linspace(1,9,100)
#
#gx, gy = np.meshgrid(temp, ratio)
#
#
#
#process_mesh= []
#for i in temp:
#    for j in ratio:
#        b = [i,j]
#        process_mesh.append(b)
#        
#process_mesh = np.array(process_mesh) 
#
#gx= process_mesh[:,0].reshape(gx.shape)
#
#gy = process_mesh[:,1].reshape(gx.shape)
#
#    1.2*0.9
#mu_process, cov_process = gpr_process.predict(process_mesh, return_cov=True)
#aaa =mu_process.reshape(gx.shape)
#mu_process[mu_process<0] = 0
#
#plt.rcParams["figure.figsize"] = [4, 3]
#plt.rcParams.update({'font.size': 22})
#plt.rcParams['xtick.major.pad'] = 8
#plt.rcParams['ytick.major.pad'] = 8
#plt.contourf(gx, gy, mu_process.reshape(gx.shape),cmap='inferno')
#
#plt.xlabel('Temp [$^\circ$C]')
#plt.ylabel('Solvent Ratio')
#clb = plt.colorbar()
#
##clb.ax.set_title('$\\eta$[%]')
#clb.ax.set_title('$\\eta$[%]')
##clb.ax.set_title('R[$]')
##clb.set_label('R[$]',labelpad=-1)
#
#ind = np.unravel_index(np.argmax(mu_process.reshape(gx.shape), axis=None), mu_process.reshape(gx.shape).shape)
#
#
#
#plt.scatter(gx[ind[0],ind[1]], gy[ind[0],ind[1]], 500,marker='*',c='r',label='Best')#, c=Y_train)
#plt.legend()
#
#plt.show()
#
#
#
##BO for the next point
#
#from GPyOpt.methods import BayesianOptimization
#
#
#bds = [{'name':'X1','type':'continuous','domain':(60,140)},
#        {'name':'X2','type':'continuous','domain':(1,9)},
#        ]
#
#from numpy.random import seed
#
#seed(1)
#
#optimizer = BayesianOptimization(f= None,
#                                   domain = bds,
#                                   model_type='GP',                                       
#                                   acquisition_type ='EI',
#                                   acquisition_jitter = 0.5,
#                                   X=process_rev[:,:2],
#                                   Y=-process_rev[:,2].reshape(-1,1),
#                                   exact_feval = False,  
#                                   minimize= True)
#
#x_next = optimizer.suggest_next_locations()
#
#
#print(x_next )
#
#
###sensitivity plot
#num = 20
#par_eff = en_exp[num].reshape(1,-1)
#constant = par_eff
#sensi = np.zeros((5,2))
#JV_ = decoder_.predict(par_eff)
#for i in range(5):
#    
#    par_plus = np.copy(par_eff)
#    par_minus = np.copy(par_eff)
#    par_plus[0,i] = par_plus[0,i]*1.1
#    
#    par_minus[0,i] = par_minus[0,i]*0.9
#    JV_plus = decoder_.predict(par_plus)
#    JV_minus = decoder_.predict(par_minus)
#    sensi[i,0] = np.max(JV_plus[:,100*2:100*3]*v_sweep)/np.max(JV_[:,100*2:100*3]*v_sweep)*100-100
#    sensi[i,1] = np.max(JV_minus[:,100*2:100*3]*v_sweep)/np.max(JV_[:,100*2:100*3]*v_sweep)*100-100
#    
#index = np.argsort(sensi[sensi>0])
#negative = sensi[sensi<0]
#positive = sensi[sensi>0]
#x = ['tau','FSRV','RSRV','Rs','Rsh' ]
#x_r = []
#for i in index:
#    x_r.append(x[i])
#fig = plt.figure()
#ax = plt.subplot(111)
#ax.bar(x_r, negative[index], width=1, color='r')
#ax.bar(x_r, positive[index], width=1, color='b') 
    