import numpy as np
import os
# os.chdir("./code")
from utils import Conformal_Prediction
import argparse
import pandas as pd
from sklearn.model_selection import ParameterGrid
from scipy.stats import norm

""" Configurations of the current run """
parser = argparse.ArgumentParser('')
parser.add_argument('--task_id', type=int, default=0)
args = parser.parse_args()
task_id = args.task_id - 1 
params = {'rho': [0.005, 0.01, 0.015, 0.02, 0.025], 'xrho': [1,5,8], 'grp': [1, 2, 3, 4, 5]}
params_grid = list(ParameterGrid(params))
rho = params_grid[task_id]['rho']
xrho = params_grid[task_id]['xrho']
seed_group = params_grid[task_id]['grp']

method = "lasso"
classifier = "logistic"
dim = 50
rho_star = 0.01
amp = 15

def get_samples(n,coef):

    """
    Obtain the training set from P_{X,Y}

    INPUT:
        - dim: dimension of X
        - n: number of the training samples
        - coef: the coefficients
 
    """
    dim = np.shape(coef)[0]
    X = np.random.normal(0,1,(n,dim))
    Y = np.matmul(X, coef) + np.random.normal(0,1,n)
    return np.concatenate([X, Y.reshape(-1,1)], axis=1)

def get_shift_samples(m, coef, A, B, abs_quantile, eta): 

    """
    Create the test set from Q
    
    Input:
        - dim: the dimension of the feature
        - m: the number of the test samples
        - c: the magnitude of the coefficient \beta_j
        - eta: parameter controlling the fraction of covariate shift
    
    """
    dim = np.shape(coef)[0]
    l=[]
    mean2=np.array([eta/10, -eta/10]+([0]*(dim-2)))
    cov2=np.identity(dim)
    for _ in range(m):
        x=np.random.multivariate_normal(mean2,cov2)
        y=sample_y(x, coef, abs_quantile, A, B)
        x=list(x)
        x.append(y)
        l.append(x)
    return np.array(l)

def f(x):
    return x*np.log(x)

def invg(r,rho):
    eps=1e-10
    left=r
    right=1
    mid=(left+right)/2
    while (right-left>eps):
        ans=mid*f(r/mid)+(1-mid)*f((1-r)/(1-mid))
        if ans<=rho:
            left=mid
        else:
            right=mid
        mid=(left+right)/2
    return mid

def sample_y(x, coef, abs_quantile, A, B):
    #cov3=1-dim*c*c
    flag=1
    while flag:
        candidate_sample=np.random.normal(np.sum(x * coef), 1) 
        uniform_sample=np.random.rand(1)
        if uniform_sample < acpt_ratio(candidate_sample, x, abs_quantile, A, B, coef):
            return candidate_sample 
        
def acpt_ratio(y,x,abs_quantile,A,B,coef):
    if abs(y - np.sum(x * coef)) <= abs_quantile:
        return A
    else :
        return B

""" Parameters """
n = 1000 # number of training samples
m = 1000 # number of test samples
s = 10 # sparsity
N = 20 # number of seeds
c = amp / np.sqrt(n)
coef = np.zeros(dim)
coef[:s] = c # correlation coefficient
alpha = 0.1 # the level of the test

cdf_value=invg(1-alpha,rho_star)
#abs_quantile = np.percentile(np.abs(np.random.normal(0,1, 1000000)),cdf_value * 100)
abs_quantile = norm.ppf(1 - (1- cdf_value) / 2)
C=(alpha)/(1-invg(1-alpha,rho_star))+1e-5
A=(1-alpha)/(invg(1-alpha,rho_star)*C)
B=(alpha)/((1-invg(1-alpha,rho_star))*C)

#rho = 0.02
#obj.shiftrho - obj.rho

""" Initialization """
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


""" Main experiment """
for seed in range(N):
    this_seed = seed_start = (seed_group - 1) * 20 + seed
    np.random.seed(this_seed)
    if seed%10 == 0:
        print(seed)
    samples = get_samples(n, coef)
    obj = Conformal_Prediction(samples,alpha,rho,'kl', 'cmr')
    fit_samples = get_samples(n,coef)
    fit_shiftsamples = get_shift_samples(m, coef, A, B, abs_quantile, xrho)
    obj.initial(fit_samples[:,:-1],fit_shiftsamples[:,:-1],fit_samples[:,-1], method, classifier)
    shiftsamples=get_shift_samples(m,coef,A,B,abs_quantile,xrho)
    for type in ['0','1','2','3','4']:
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
            bool, q=obj.one_test(shiftsample,type)
            if bool:
                count+=1
            lenth+=q
        li[seed]=count/m
        length[seed]=2*lenth/m

coverage = np.transpose(np.array([li0,li1,li2,li3,li4]))
lens = np.transpose(np.array([len0,len1,len2,len3,len4]))

"""Save the output"""
coverage = pd.DataFrame(coverage)
lens = pd.DataFrame(lens)

set_name = 'rho_star_' + str(rho_star*1000) + '_xrho_' + str(xrho) + '_rho_' + str(rho*1000) + '_amp_' + str(amp) + '_d_' + str(dim) + '_grp_' + str(seed_group)

coverage.to_csv('../results/' + set_name + '_cov.csv')
lens.to_csv('../results/' + set_name +  '_lens.csv')

