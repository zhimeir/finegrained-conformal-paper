import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
#import warnings
#import os 
import argparse
#os.chdir("./code")
#from qosa import base_forest
from utils import Conformal_Prediction
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#warnings.filterwarnings("ignore", category=FutureWarning, message="`max_features='auto'` has been deprecated*")

""" Configurations of the current run """
parser = argparse.ArgumentParser('')
parser.add_argument('--task_id', type = int, default = 1)
args = parser.parse_args()
task_id = args.task_id - 1 
params = {'rho': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04], 'grp': range(1, 21)}
params_grid = list(ParameterGrid(params))
rho = params_grid[task_id]['rho']
seed_group = params_grid[task_id]['grp']


""" Parameters """
alpha = 0.2
N = 5 ## number of seeds
df = pd.read_csv('../datasets/semi_synthetic_data.csv')
observe_feature = ['S3', 'C1', 'C2', 'C3', 'X3', 'X4', 'X5', 'Y1'] #, 'XC', 'X1', 'X2']
treated = df[df['Z']==1]
untreated = df[df['Z']==0]
all_samples = treated[observe_feature].values
all_shiftsamples = untreated[observe_feature].values
dim = len(observe_feature)-1
coverage = []
lens = []


li0 = [0] * N
li1 = [0] * N
li2 = [0] * N
li3 = [0] * N
li4 = [0] * N
len0 = [0] * N
len1 = [0] * N
len2 = [0] * N
len3 = [0] * N
len4 = [0] * N

# n = 1500 
# m = 1000

for seed in range(N):

    this_seed = seed_start = (seed_group - 1) * N + seed
    np.random.seed(this_seed)
    if seed%10 == 0:
        print(seed)
    
    X=all_samples[:,:dim]
    y=all_samples[:,dim]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.5, random_state = this_seed)
    samples = np.concatenate([X_train,y_train.reshape(-1,1)],axis=1)
    #train_row_index = np.random.choice(len(samples), size=n, replace=False)
    #train_data_ = samples[train_row_index,:]
    #obj = Conformal_Prediction(train_data_, alpha, rho, 'chi_square', "cmr")
    obj = Conformal_Prediction(samples, alpha, rho, 'chi_square', "cmr")
    samples = np.concatenate([X_test,y_test.reshape(-1,1)],axis=1)
    X = all_shiftsamples[:,:dim]
    y = all_shiftsamples[:,dim]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.5, random_state = this_seed)
    shiftsamples = np.concatenate([X_test,y_test.reshape(-1,1)],axis=1)
    obj.initial(samples[:,:-1],shiftsamples[:,:-1],samples[:,-1],'random_forest', 'random_forest')
    shiftsamples=np.concatenate([X_train,y_train.reshape(-1,1)],axis=1)
    #shift_row_index=np.random.choice(len(shiftsamples),size=m,replace=False)
    #shift_data_=shiftsamples[shift_row_index]
    for type in ['0', '1', '2', '3', '4']:
        if type=='0':
            li=li0
            length=len0
        if type=='1':
            li=li1
            length=len1
        if type=='2':
            li=li2
            length=len2
        if type=='3':
            li=li3
            length=len3
        if type=='4':
            li=li4
            length=len4
        count=0
        lenth=0
        for shiftsample in shiftsamples:
        ## for shiftsample in shift_data_:
            bool, quant = obj.one_test(shiftsample,type)
            if bool:
                count+=1
            lenth += quant
        li[seed]=count/shiftsamples.shape[0]
        length[seed]=2*lenth/shiftsamples.shape[0]
        ##li[seed]=count/m
        ##length[seed]=2*lenth/m

coverage = np.transpose(np.array([li0,li1,li2,li3,li4]))
lens = np.transpose(np.array([len0,len1,len2,len3,len4]))

"""Save the output"""
coverage = pd.DataFrame(coverage)
lens = pd.DataFrame(lens)

set_name = 'nslm_XCX1X2_rho_' + str(rho*1000) + '_grp_' + str(seed_group)
coverage.to_csv('../results/' + set_name + '_cov.csv')
lens.to_csv('../results/' + set_name +  '_lens.csv')


""" Diagnostics """
"""
mdl = RandomForestClassifier()
merged_X=np.concatenate([all_samples[:,:-1],all_shiftsamples[:,:-1]],axis=0)
label0=np.zeros(all_samples.shape[0])
label1=np.ones(all_shiftsamples.shape[0])
P0 = all_samples.shape[0]
P1 = all_shiftsamples.shape[0]
merged_label=np.concatenate([label0,label1])
mdl.fit(merged_X,merged_label)
pr = mdl.predict_proba(merged_X)
rho_x = np.mean(f_chi((P0/P1)*(pr[:P0,1]/pr[:P0,0])).mean())

mdl1 = RandomForestClassifier()
merged_all=np.concatenate([all_samples,all_shiftsamples],axis=0)
mdl1.fit(merged_all,merged_label)
pr = mdl1.predict_proba(merged_all)
rho_all = np.mean(f_chi((P0/P1)*(pr[:P0,1]/pr[:P0,0])).mean())
rho_all 
rho_x

np.mean(mdl.predict(merged_X) != merged_label)
np.sum(mdl.predict(merged_X))

mdl = RandomForestRegressor()
mdl.fit(all_samples[:,:-1],all_samples[:,-1])
res = np.abs(mdl.predict(all_samples[:,:-1]) - all_samples[:,-1])

#mdl = RandomForestRegressor()
#mdl.fit(all_shiftsamples[:,:-1],all_shiftsamples[:,-1])
sres = np.abs(mdl.predict(all_shiftsamples[:,:-1]) - all_shiftsamples[:,-1])

np.quantile(res,0.9)
np.quantile(sres,0.8)
"""
