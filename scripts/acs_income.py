from whyshift import get_data
import pandas as pd
import numpy as np
import os
# os.chdir("./code")
from sklearn.model_selection import train_test_split, ParameterGrid
import warnings
from utils import Conformal_Prediction
import argparse
# warnings.filterwarnings("ignore", category=FutureWarning, message="`max_features='auto'` has been deprecated*")

""" Configurations of the current run """
parser = argparse.ArgumentParser('')
parser.add_argument('--task_id', type=int, default=1)
args = parser.parse_args()
task_id = args.task_id - 1 
params = {'rho': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04], 'grp': range(1,11)}
params_grid = list(ParameterGrid(params))
rho = params_grid[task_id]['rho']
seed_group = params_grid[task_id]['grp']

""" Load dataset """
X0, y0, feature_names = get_data("income", "NY", False, '../datasets/acs/', 2018)
X1, y1, feature_names = get_data("income", "SD", False, '../datasets/acs/', 2018)
all_samples = np.concatenate([X0,y0.reshape(-1,1)],axis=1)
all_shiftsamples = np.concatenate([X1,y1.reshape(-1,1)],axis=1)
dim = X0.shape[1]





""" Parameters """
n = 1000 # training sample size
m = 1000 # test sample size
alpha = 0.2 # level of the test
N = 10 # number of the seeds

""" Initialization """
coverage = []
lens = []
li0 = [0] * N
li1 = [0] * N
li2 = [0] * N
li3 = [0] * N
li4 = [0] * N
length0 = [0] * N
length1 = [0] * N
length2 = [0] * N
length3 = [0] * N
length4 = [0] * N

""" Main experiment """
for seed in range(N):
    this_seed = seed_start = (seed_group - 1) * N + seed
    np.random.seed(this_seed)
    if seed % 10 == 0:
        print(seed)
    X = all_samples[:,:dim]
    y = all_samples[:,dim]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = n, random_state = this_seed)
    samples = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1) # whole training data
    train_row_index = np.random.choice(np.shape(samples)[0], size = n, replace=False) 
    train_data_=samples[train_row_index,:] # randomly select part of training data to construct prediction interval
    obj = Conformal_Prediction(train_data_, alpha, rho, 'chi_square', 'aps')
    samples = np.concatenate([X_test, y_test.reshape(-1,1)], axis=1) # calibration data for weight function, etc
    X = all_shiftsamples[:,:dim]
    y = all_shiftsamples[:,dim]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=m, random_state = this_seed)
    shiftsamples = np.concatenate([X_train,y_train.reshape(-1,1)],axis=1) # calibration data for weight function, etc
    obj.initial(samples[:,:-1],shiftsamples[:,:-1],samples[:,-1],'random_forest_classifier', 'xgb')
    shiftsamples=np.concatenate([X_test,y_test.reshape(-1,1)],axis=1)
    shift_row_index=np.random.choice(np.shape(shiftsamples)[0],size=m,replace=False)
    shift_data_=shiftsamples[shift_row_index] # randomly select part of test data to test prediction interval
    for type in ['0', '1', '2', '3', '4']:
        if type=='0':
            li=li0
            length=length0
        if type=='1':
            li=li1
            length=length1
        if type=='2':
            li=li2
            length=length2
        if type=='3':
            li=li3
            length=length3
        if type=='4':
            li=li4
            length=length4
        count = 0
        lens = 0
        for shiftsample in shift_data_:
            bool, len=obj.one_test(shiftsample,type)
            if bool:
                count+=1
            lens += len
        li[seed]=count/m
        length[seed] = lens/m

coverage = np.transpose(np.array([li0,li1,li2,li3,li4]))
lens = np.transpose(np.array([length0,length1,length2,length3,length4]))

"""Save the output"""
coverage = pd.DataFrame(coverage)
lens = pd.DataFrame(lens)

set_name = 'acsincome_ny_sd_rho_' + str(rho*1000) + '_grp_' + str(seed_group)
coverage.to_csv('../results/' + set_name + '_cov.csv')
lens.to_csv('../results/' + set_name +  '_lens.csv')

