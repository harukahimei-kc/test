#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import sys
import itertools
import os
import datetime
import shutil
import warnings

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from sklearn import model_selection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel
plt.rcParams['font.family'] = 'MS Gothic'


# In[2]:


warnings.simplefilter('ignore')


# #### Load data

# In[3]:


print('Loading data.   Stage 1/5')


# In[4]:


All_Files = glob.glob('C:\DoE_data\*.csv')


# In[5]:


pre_exam = [s for s in All_Files if ('Mokuteki_settings' not in s) and ('Setsumei_settings' not in s)][0]


# In[6]:


dep_set = [s for s in All_Files if 'Mokuteki_settings' in s][0]
ind_set = [s for s in All_Files if 'Setsumei_settings' in s][0]


# In[7]:


df_pre = pd.read_csv(pre_exam, encoding='cp932')
df_dep = pd.read_csv(dep_set, encoding='cp932', index_col=0)
df_ind = pd.read_csv(ind_set, encoding='cp932', index_col=0)


# #### Make experiment parameters

# In[8]:


print('Making experiment parameters.   Stage 2/5')
print('It might take a little long time.')


# In[9]:


cand_col = list(df_ind.columns)
uncommon_cat = 0
drop_col = []
add_list = []
target_col = list(df_dep.columns)


# In[10]:


for item in cand_col:
    # Num or Bool or Category
    data_type = int(df_ind.loc['Num_Bool_Cat', item])
    
    # Num
    if data_type==0:
        set_min = df_ind.loc['Min', item]
        set_max = df_ind.loc['Max', item]
        set_interval = df_ind.loc['interval', item]
        # float or not
        if df_ind[item].dtype=='float':
            # max 
            set_min_nd = len(str(set_min).split('.')[1])
            set_max_nd = len(str(set_max).split('.')[1])
            set_int_nd = len(str(set_interval).split('.')[1])
            nd_max = max(set_min_nd, set_max_nd, set_int_nd)
            add_list.append([s/(10**nd_max) for s in range(int(set_min*10**nd_max),
                                                  int(set_max*10**nd_max+set_interval*10**nd_max),
                                                  int(set_interval*10**nd_max))])

        else:
            add_list.append(list(range(set_min, set_max+set_interval, set_interval)))
            
    # Bool
    elif data_type==1:
        add_list.append([0,1])
        
    # Category
    elif data_type==2:
        pre_cat = list(df_pre[item].unique())
        cand_cat = list(df_ind.loc['interval', item].split(','))
        for cat in cand_cat:
            if cat not in pre_cat:
                uncommon_cat = uncommon_cat+1
                
        # If there are new categories, drop columns
        if uncommon_cat>0:
                df_pre = df_pre.drop(item, axis=1)
                df_ind = df_ind.drop(item, axis=1)
                drop_col.append(item)
                print('Attention!!')
                print('Categorical column ' + item + ' was deleted.')
        
        # If all of categories were included in pre-exam
        else:
            add_list.append(cand_cat)
            
    # If there are wrong identifier, print message and exit process
    else:
        print('Data type is wrong. Please check Num_Bool_Cat in Setsumei_setting.csv')
        sys.exit()


# In[11]:


cand_list = []
cand_col = list(df_ind.columns)


# In[12]:


for element in itertools.product(*add_list):
    cand_list.append(element)

if len(cand_list)>5000000:
    print('The number of experiment is over 5000000. It might cause Memory error.')

df_cand = pd.DataFrame(cand_list, columns=cand_col)


# In[13]:


del cand_list


# #### One-Hot encoding

# In[14]:


cate_list = list(df_ind.columns[df_ind.loc['Num_Bool_Cat', :].astype(int)==2])


# In[15]:


for item in cate_list:
    pre_cat = list(df_pre[item].unique())
    cand_cat = list(df_ind.loc['interval', item].split(','))
    
    pre_cat.sort()
    cand_cat.sort()
    
    if pre_cat[0]!=cand_cat[0]:
        df_cand = pd.get_dummies(df_cand, columns=[item], drop_first=False)
        
    else:
        df_cand = pd.get_dummies(df_cand, columns=[item], drop_first=True)


# In[16]:


df_pre = pd.get_dummies(df_pre, drop_first=True)


# #### GPR and Bayesian Optimization

# In[17]:


print('Simulation start.   Stage 3/5')


# In[18]:


dt_now = datetime.datetime.now().strftime('%y%m%d%H%M')


# ##### Make a folder for results

# In[19]:


new_dir = 'C:\DoE_data\DoE_data_' + dt_now
os.makedirs(new_dir, exist_ok=True)


# ##### Dropped columns list

# In[20]:


if len(drop_col)>0:
    params_txt = 'Drop_columns_' + dt_now + '.txt'
    save_dir = os.path.join(new_dir,params_txt)

    with open(save_dir, mode='w') as f:
        print(drop_col, file=f)


# In[21]:


fold_number = 10
epsilon = 0.01

# Number of x, y
number_of_y_variables = len(df_dep.columns)
n_of_x = df_cand.shape[1]

# Standardization (not use preprocessing)
x = df_pre[list(df_cand.columns)]
y = df_pre.drop(list(df_cand.columns), axis=1)
sc_x = (x - x.mean()) / x.std()
sc_x_for_prediction = (df_cand - x.mean()) / x.std()
sc_y = (y - y.mean()) / y.std()
mean_of_y = y.mean()
std_of_y = y.std()

est_y = np.zeros([df_cand.shape[0], number_of_y_variables])
std_y_for_prediction = np.zeros([df_cand.shape[0], number_of_y_variables])
plt.rcParams['font.size'] = 14
for y_number in range(number_of_y_variables):
    model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(), alpha=0)
    model.fit(sc_x, sc_y.iloc[:, y_number])
    
    y_temp, std_temp = model.predict(
        sc_x_for_prediction, return_std=True)
    est_y[:, y_number] = y_temp
    std_y_for_prediction[:, y_number] = std_temp

    estimated_y, sigma = model.predict(sc_x, return_std=True)
    estimated_y = estimated_y * std_of_y.iloc[y_number] + mean_of_y.iloc[y_number]
    
    plt.figure(figsize=(6,6))
    plt.scatter(y.iloc[:, y_number], estimated_y)
    y_max = max(y.iloc[:, y_number].max(), estimated_y.max())
    y_min = min(y.iloc[:, y_number].min(), estimated_y.min())
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('actual y')
    plt.ylabel('estimated y')
    
    
    plt.figure(figsize=(6,6))
    plt.plot(x.iloc[:,0], y.iloc[:,y_number], 'r.', markersize=10, label='Observations')
    plt.errorbar(x.iloc[:,0], estimated_y, yerr=sigma*1.96, fmt='b.', markersize=10,
                 label='Prediction and 95% confidence interval')
    

    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.xlabel(x.columns[0])
    plt.ylabel(y.columns[y_number])
    sct_file = str(x.columns[0]) + 'と' + str(y.columns[y_number]) + '予実散布図_' + dt_now + '.jpeg'
    save_dir = os.path.join(new_dir, sct_file)
    plt.savefig(save_dir, bbox_inches='tight')
    
    estimated_y_in_cv = model_selection.cross_val_predict(model, sc_x, sc_y.iloc[:, y_number],
                                                          cv=fold_number)
    estimated_y_in_cv = estimated_y_in_cv * std_of_y.iloc[y_number] + mean_of_y.iloc[y_number]
    
    plt.figure(figsize=(6,6))
    plt.scatter(y.iloc[:, y_number], estimated_y_in_cv)
    y_max = max(y.iloc[:, y_number].max(), estimated_y_in_cv.max())
    y_min = min(y.iloc[:, y_number].min(), estimated_y_in_cv.min())
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel(str(y.columns[y_number])+'実測値')
    plt.ylabel(str(y.columns[y_number])+'予測値')
    plt.title(str(y.columns[y_number])+'シミュレーションモデルの精度')
    acc_file = str(y.columns[y_number]) + 'モデル精度_' + dt_now + '.jpeg'
    save_dir = os.path.join(new_dir, acc_file)
    plt.savefig(save_dir, bbox_inches='tight')
    

est_y = pd.DataFrame(est_y)
est_y.columns = y.columns
est_y = est_y * y.std() + y.mean()
std_y_for_prediction = pd.DataFrame(std_y_for_prediction)
std_y_for_prediction.columns = y.columns
std_y_for_prediction = std_y_for_prediction * y.std()

# Probabilities
probabilities = np.zeros(est_y.shape)
for y_number in range(number_of_y_variables):
    # Maimization
    if df_dep.iloc[0, y_number] == 1:
        probabilities[:, y_number] = 1 - norm.cdf(max(y.iloc[:, y_number]) + std_of_y.iloc[y_number] * epsilon,
                                                  loc=est_y.iloc[:, y_number],
                                                  scale=std_y_for_prediction.iloc[:, y_number])
    # Minimization
    elif df_dep.iloc[0, y_number] == -1:
        probabilities[:, y_number] = norm.cdf(min(y.iloc[:, y_number]) - std_of_y.iloc[y_number] * epsilon,
                                              loc=est_y.iloc[:, y_number],
                                              scale=std_y_for_prediction.iloc[:, y_number])

    # Fit a range
    elif df_dep.iloc[0, y_number] == 0:
        probabilities[:, y_number] = norm.cdf(df_dep.iloc[2, y_number],
                                              loc=est_y.iloc[:, y_number],
                                              scale=std_y_for_prediction.iloc[:, y_number]) - norm.cdf(
            df_dep.iloc[1, y_number],
            loc=est_y.iloc[:, y_number],
            scale=std_y_for_prediction.iloc[:, y_number])

    probabilities[std_y_for_prediction.iloc[:, y_number] <= 0, y_number] = 0


# ####  Delete needless data

# In[22]:


del df_pre, df_dep, x, sc_x, sc_x_for_prediction, sc_y, std_y_for_prediction, y_temp, std_temp, estimated_y, estimated_y_in_cv


# #### Save results

# In[23]:


print('Saving results.   Stage 4/5')
print('It might take a little long time.')


# ##### csv, txt

# In[24]:


# Each parameters and predictions
pred_v_file = 'all_experiments_and_predicted_values_' + dt_now + '.csv'
save_dir = os.path.join(new_dir, pred_v_file)
df_res = pd.concat([df_cand, est_y], axis=1)
df_res.to_csv(save_dir, encoding='cp932')

del est_y
    
# Save each probabilities
prob_file = 'each_probabilities_' + dt_now + '.csv'
save_dir = os.path.join(new_dir, prob_file)
probabilities = pd.DataFrame(probabilities)
probabilities.columns = y.columns
probabilities.index = df_cand.index
probabilities.to_csv(save_dir, encoding='cp932')

# Save sum of predictions
sum_of_prob_file = 'sum_of_log_probabilities_' + dt_now + '.csv'
save_dir = os.path.join(new_dir, sum_of_prob_file)

sum_of_log_probabilities = (np.log(probabilities)).sum(axis=1)
sum_of_log_probabilities = pd.DataFrame(sum_of_log_probabilities)
# replace log0(-inf)
sum_of_log_probabilities[sum_of_log_probabilities == -np.inf] = -10 ** 100
sum_of_log_probabilities.columns = ['sum_of_log_probabilities']
sum_of_log_probabilities.index = df_cand.index
sum_of_log_probabilities.to_csv(save_dir, encoding='cp932')

del probabilities

# Next candidend ID
cand_id = sum_of_log_probabilities.iloc[:, 0].idxmax()

params_txt = 'Next_experimtent_' + dt_now + '.txt'
save_dir = os.path.join(new_dir,params_txt)

with open(save_dir, mode='w') as f:
    print('candidate id: {}'.format(cand_id), file=f)
    print(df_cand.loc[cand_id], file=f)


# ##### Save plots

# In[25]:


for item in target_col:

    plt.figure(figsize=(20,8))

    plt.plot(df_res.loc[:,item])
    plt.axvline(x=cand_id, color='r')

    plt.ticklabel_format(style='plain',axis='both')
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

    plt.title(item)
    plt.xlabel('実験ID')
    
    s_file = str(item) + '全実験条件予測結果_' + dt_now + '.jpeg'
    save_dir = os.path.join(new_dir, s_file)
    plt.savefig(save_dir, bbox_inches='tight')


# In[26]:


del df_cand


# #### Move input csv

# In[27]:


# pre-exam
shutil.move(pre_exam, new_dir)
# Dependent variables
shutil.move(dep_set, new_dir)
# Independent variables
shutil.move(ind_set, new_dir)


# In[28]:


print('Completed all processes.   Stage 5/5')

