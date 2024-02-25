import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  
from sklearn.linear_model import LinearRegression,LassoCV,LogisticRegressionCV
from pygam import LinearGAM, s
from qosa import base_forest
import warnings
from scipy.stats import norm
warnings.filterwarnings("ignore", category=FutureWarning, message="`max_features='auto'` has been deprecated*")

class Conformal_Prediction:
    def __init__(self, samples, alpha, rho, f_type, score_type = 'cmr'):
        """
        Initializes a new instance of the Conformal_Prediction.

        Args:
        samples: the training sample from P
        alpha: the level of prediction interval
        rho: parameter to construct interval
        f_type: the choice of f divergence
                it should be chosen from ['kl','tv','chi_square']
                - 'kl': KL divergence
                - 'tv': Total variation distance
                - 'chi_square': Chi-square divergence
        score_type: the choice of score function
                    it should be chosen from ['cmr','aps']
                    - 'cmr': Conformal mean regression
                    - 'aps': APS
        """
        self.sample = samples
        self.alpha = alpha
        self.rho = rho
        self.score_type = score_type
        self.f_type = f_type

        if f_type == 'kl':
            self.f = f_kl
        elif f_type == 'tv':
            self.f = f_tv
        elif f_type == 'chi_square':
            self.f = f_chi

        if score_type == 'cmr':
            self.score = cmr
            self.smax = np.inf
        if score_type == 'aps':
            self.score = aps 
            self.smax = 1


    def initial(self, X, shift_X, y, model_type, classifier_type, tree_depth = None):
        """
        fit weight function, outcome function and conditional cumulative distribution function

        Args:
        X: the training sample from P
        shift_X: the training sample from Q
        y: the training sample from P
        model_type: the type of the outcome model to be fitted
                    - 'linear','GAM','lasso','logistic','random_forest','random_forest_classifier'
        classifier_type: the type of the classifier to estimate covariate shift
                    - 'logistic','random_forest','xgb'
        """
        self.get_w(X, shift_X, classifier_type, tree_depth)
        print("w done.")
        self.get_model(X, y, model_type)
        print("u done.")
        self.get_m(X, y)

    def get_w(self,X,shift_X, classifier_type='logistic', tree_depth = None):
        """
        get the weight function
        X: the training sample from P
        shift_X: the training sample from Q 
        classifier_type: the type of the classifier to estimate covariate shift
                    - 'logistic','random_forest','xgb'
        """
        if classifier_type == 'logistic':
            mdl_classifier = LogisticRegressionCV(penalty='l1', solver = 'liblinear')
        if classifier_type == 'random_forest':
            mdl_classifier = RandomForestClassifier(max_depth = tree_depth)
        if classifier_type == 'xgb':
            from whyshift import fetch_model
            mdl_classifier = fetch_model('xgb')

        P0 = X.shape[0]
        P1 = shift_X.shape[0]
        label0=np.zeros(P0)
        label1=np.ones(P1)
        merged_X=np.concatenate([X,shift_X],axis=0)
        merged_label=np.concatenate([label0,label1])
        #rf_classifier.fit(merged_X,merged_label)
        mdl_classifier.fit(merged_X,merged_label)
        #print(mdl_classifier.coef_)
        X_dim=self.sample.shape[1]-1
        sample_X=self.sample[:,:X_dim]
        def w(x):
            """
            get the estimate weight of training sample X and test sample x
            """
            X_all=np.concatenate([sample_X,x],axis=0)
            p = mdl_classifier.predict_proba(X_all)
            p = np.maximum(p, 1e-10) # avoid division by zero
            
            return (P0/P1)*(p[:,1]/p[:,0])
        self.w=w
        p = mdl_classifier.predict_proba(merged_X)
        p = np.maximum(p, 1e-10) # avoid division by zero
        if(self.f_type == 'kl'):
            self.shiftrho = self.rho + np.log((P0/P1)*(p[P0:(P0+P1),1]/p[P0:(P0+P1),0])).mean()  #estimate parameter rho for robust CP
        self.shiftx=shift_X

    def get_model(self,X,y,model_type):
        """
        fitting the outcome model
        samples: the training sample from P 
        model_type: the type of the outcome model to be fitted
        """
        if model_type=='linear':
            model=LinearRegression()
        elif model_type=='GAM':
            model=LinearGAM()
        elif model_type=='lasso':
            model=LassoCV()
        elif model_type == 'logistic':
            model = LogisticRegressionCV(penalty='l1', solver = 'liblinear')
        elif model_type == 'random_forest':
            model = RandomForestRegressor()
        elif model_type == 'random_forest_classifier':
            model = RandomForestClassifier()
        model.fit(X,y)
        self.model_u=model
    
    def get_m(self, X_train, Y_train):
        """
        fitting s(X,Y)|X
        
        INPUT:
        
            - X_train, Y_train: the training sample from P
        """        
        
        U=np.random.uniform(size = np.shape(X_train)[0])
        gp = base_forest.QuantileRegressionForest()
        Y_train = self.score(X_train,Y_train,self.model_u, U)
        gp.fit(X_train,Y_train)
        y_list = np.sort(Y_train)
        N = X_train.shape[0] - 1
        def estimate_conditional_cdf(x_values,t_value):    
            C_CDF = gp.predict_C_CDF(x_values)
            l = 0
            r = N
            while r-l>1:
                mid = (l+r)//2
                if y_list[mid]<=t_value:
                    l=mid
                else:
                    r=mid
            return C_CDF[:,mid]
        self.m = estimate_conditional_cdf
        samples = self.sample
        X_dim = samples.shape[1]-1
        X=samples[:,:X_dim]
        Y=samples[:,X_dim]
        U = np.random.uniform(size = np.shape(X)[0])
        Ss = self.score(X,Y,self.model_u,U)
        Weights=self.w([X[0]])
        Weights=Weights[:-1]
        self.q=dr_quant(Ss,Weights,self.m,X,self.shiftx,self.invg(1-self.alpha,self.rho))
        self.u = U
 

    # inverse of g_{f,rho} 
    def invg(self,r,rho):
        f=self.f
        eps=1e-10
        if r>1:
            return 1
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
    
    def get_interval(self,shift_X,type):

        """ 
        Construct conformal prediction intervals 
        INPUT:
            - shift_X: the test sample from Q
        """

        samples = self.sample
        model_u = self.model_u
        X_dim = samples.shape[1]-1
        X = samples[:,:X_dim]
        Y = samples[:,X_dim]
        Ss = self.score(X,Y,model_u,self.u)
        if type=='0': # standard CP

            Ss=np.concatenate([Ss, np.array([self.smax])])
            weq=np.ones_like(Ss)
            quantile=quantile_weighted(Ss,weq,1-self.alpha)

        elif type=='1': # weighted CP

            Weights=self.w(shift_X) 
            Weights=Weights/np.sum(Weights)
            Ss=np.concatenate([Ss, np.array([self.smax])])
            quantile=quantile_weighted(Ss,Weights,1-self.alpha)

        elif type=='2': # robust CP

            Ss=np.concatenate([Ss, np.array([self.smax])])
            weq=np.ones_like(Ss)
            quantile=quantile_weighted(Ss,weq,self.invg(1-self.alpha,self.shiftrho))

        elif type=='3': #weighted robust CP

            Weights=self.w(shift_X) 
            Weights=Weights/np.sum(Weights)
            Ss=np.concatenate([Ss, np.array([self.smax])])
            quantile=quantile_weighted(Ss,Weights,self.invg(1-self.alpha,self.rho))

        elif type=='4': #DR weighted robust CP
            
            quantile = self.q

        return quantile
    
    
    
    def one_test(self,shift_sample,type):
        """
        test prediction interval for one test sample

        Args:
        shift_sample: test sample
        type: the type of prediction interval
            - '0': standard CP
            - '1': weighted CP
            - '2': robust CP
            - '3': weighted robust CP
            - '4': DR weighted robust CP
        """
        shift_X = np.expand_dims(shift_sample[:-1],axis=0)
        shift_Y = shift_sample[-1]
        quant = self.get_interval(shift_X,type)
        U = np.random.uniform()
        new_score = self.score(shift_X,shift_Y,self.model_u,U)
        if(self.score_type == 'aps'):
            new_score = new_score[0]

        if new_score <= quant: # check if the true outcome is within the interval
            if_cover = 1
        else:   
            if_cover = 0

        if self.score_type == 'aps':
            lens = (self.score(shift_X,0,self.model_u,U)[0] <= quant) + (self.score(shift_X,1,self.model_u,U)[0] <= quant)
        else:
            lens = 2 * quant 
        return if_cover, lens


def f_kl(x):
    """ KL divergence """
    return x*np.log(x)

def f_tv(x):
    """Total variation distance"""
    return np.abs(x-1)/2

def f_chi(x):
    """ Chi-square divergence"""
    return (x-1)**2

def cmr(x,y,mdl, u = None):
    """ Conformal mean regression """
    return np.abs(y-mdl.predict(x))

def aps(x,y,mdl,u):
    """ APS """
    p=mdl.predict_proba(x)
    return (p[:,0]>p[:,1])*((1-y)*(1-u)*p[:,0]+y*(1-u*p[:,1]))+(p[:,0]<=p[:,1])*(y*(1-u)*p[:,1]+(1-y)*(1-u*p[:,0]))


def quantile_weighted(seq, wseq, beta):

    """
        Compute weighted quantiles
    """
    sorted_indices = np.argsort(seq)
    sorted_seq = seq[sorted_indices]
    sorted_wseq = wseq[sorted_indices]
    cumulative_weights = np.cumsum(sorted_wseq)
    desired_index = np.searchsorted(cumulative_weights, beta * cumulative_weights[-1])
    quantile = sorted_seq[desired_index]
    return quantile.item()


def dr_quant(Ss,w,m,x,shiftx,q):
    """
        Compute quantiles in DR weighted robust CP
    """
    def est_cov(t):
        if sum(w) <= 1e-5:
            return np.sum(m(shiftx,t))/shiftx.shape[0]
        else:
            return np.sum(w*((Ss<=t)-m(x,t)))/np.sum(w)+np.sum(m(shiftx,t))/shiftx.shape[0]

    t_list=np.linspace(max(Ss),0,1000)
    inf_p=2
    t_pre=t_list[0]
    for t in t_list:
        inf_p=min(inf_p,est_cov(t))
        if inf_p<q:
            return t_pre
        t_pre=t
    return t_pre
        
        
    
