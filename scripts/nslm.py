import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
import argparse
from utils import Conformal_Prediction
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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
alpha = 0.2 # level of the test
N = 5 ## number of seeds
df = pd.read_csv('../datasets/nslm_semi_synthetic.csv')
observe_feature = ['S3', 'C1', 'C2', 'C3', 'X3', 'X4', 'X5', 'Y1'] #, 'XC', 'X1', 'X2']
treated = df[df['Z']==1]
untreated = df[df['Z']==0]
all_samples = treated[observe_feature].values
all_shiftsamples = untreated[observe_feature].values
dim = len(observe_feature)-1


""" Initializations """
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
    obj = Conformal_Prediction(samples, alpha, rho, 'chi_square', "cmr")
    samples = np.concatenate([X_test,y_test.reshape(-1,1)],axis=1)
    X = all_shiftsamples[:,:dim]
    y = all_shiftsamples[:,dim]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.5, random_state = this_seed)
    shiftsamples = np.concatenate([X_test,y_test.reshape(-1,1)],axis=1)
    obj.initial(samples[:,:-1],shiftsamples[:,:-1],samples[:,-1],'random_forest', 'random_forest')
    shiftsamples=np.concatenate([X_train,y_train.reshape(-1,1)],axis=1)
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
            bool, quant = obj.one_test(shiftsample,type)
            if bool:
                count+=1
            lenth += quant
        li[seed]=count/shiftsamples.shape[0]
        length[seed]=2*lenth/shiftsamples.shape[0]

coverage = np.transpose(np.array([li0,li1,li2,li3,li4]))
lens = np.transpose(np.array([len0,len1,len2,len3,len4]))

"""Save the output"""
coverage = pd.DataFrame(coverage)
lens = pd.DataFrame(lens)

set_name = 'nslm_XCX1X2_rho_' + str(rho*1000) + '_grp_' + str(seed_group)
coverage.to_csv('../results/' + set_name + '_cov.csv')
lens.to_csv('../results/' + set_name +  '_lens.csv')


