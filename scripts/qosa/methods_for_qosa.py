# -*- coding: utf-8 -*-

"""
Add docstring of the module
"""


import numpy as np
from sklearn.utils import check_random_state


__all__ = ['qosa_Quantile__Averaged_Quantile', 'qosa_Quantile__Kernel_CDF',
           'qosa_Quantile__Weighted_CDF', 'qosa_Min__Kernel_Min', 
           'qosa_Min__Min_in_Leaves', 'qosa_Min__Weighted_Min', 
           'qosa_Min__Weighted_Min_with_complete_forest']


class DefaultParametersForForestMethod(object):
    """
    Basic default parameters for all methods based on the Random Forest method.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 alpha,
                 n_estimators=100,
                 min_samples_leaf=20,
                 used_bootstrap_samples=False,
                 optim_by_CV=False,
                 n_fold=3,
                 random_state_Forest=None):

        if isinstance(alpha, (int, np.integer, float, np.floating)):
           alpha = [alpha]
        assert isinstance(n_estimators, (int, np.integer)), \
            "The number of tree should be an integer."
        assert n_estimators > 0, \
            "The number of tree should be positive: %d<0" % (n_estimators)
        assert isinstance(min_samples_leaf, (int, np.integer, np.ndarray)), \
            "The minimum number of samples required to be a leaf node \
            should be an integer or a ndarray."
        if isinstance(min_samples_leaf, (int, np.integer)):
            assert min_samples_leaf > 0, \
                "The minimum number of samples required to be leaf node should be \
                positive: %d<0" % (min_samples_leaf)
        assert isinstance(optim_by_CV, bool), \
            "The parameter for the cross validation should be a boolean."
        assert isinstance(n_fold, (int, np.integer)), \
            "The number of folds should be an integer."
        assert n_fold > 0, \
            "The number of folds should be positive: %d<0" % (n_fold)
        if isinstance(min_samples_leaf, (int, np.integer)) and optim_by_CV==True:
            raise ValueError("Optimization by CV is useless on one value."
                             "You should change by 'optim_by_CV=False'.")
        if isinstance(min_samples_leaf, np.ndarray) and optim_by_CV==False:
            raise ValueError("Optimization with the CV has to be True if "
                              "you give a ndarray for 'min_samples_leaf'.")
        assert isinstance(used_bootstrap_samples, bool), \
            "The parameter 'used_bootstrap_samples' should be a boolean."
        
        self.alpha = np.asarray(alpha)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.used_bootstrap_samples = used_bootstrap_samples
        self.optim_by_CV = optim_by_CV
        self.n_fold = n_fold
        self.random_state_Forest = check_random_state(random_state_Forest)

    @property
    def CV_strategy(self):
        return self._CV_strategy
    
    @property
    def name(self):
        return self._name


class DefaultParametersForKernelMethod(object):
    """
    Basic default parameters for all Kernel methods.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 alpha,
                 bandwidth=None,
                 optim_by_CV=False,
                 n_fold=3):

        if isinstance(alpha, (int, np.integer, float, np.floating)):
           alpha = [alpha]
        if bandwidth is not None and not isinstance(bandwidth, np.ndarray):
            assert isinstance(bandwidth, (float, np.floating)), \
                "The 'bandwidth' parameter should be a float."
            assert bandwidth>0, "The bandwidth should be positive: %f<0" % (bandwidth,)
        assert isinstance(optim_by_CV, bool), \
            "The parameter for the cross validation should be a boolean."
        if isinstance(bandwidth, (float, np.floating, type(None))) and optim_by_CV==True:
            raise ValueError("Optimization by CV is useless on one value."
                             "You should change 'optim_by_CV=False'.")
        if isinstance(bandwidth, np.ndarray) and optim_by_CV==False:
            raise ValueError("Optimization with the CV has to be True if "
                              "you give a ndarray for 'bandwidth'.")
        assert isinstance(n_fold, (int, np.integer)), \
            "The number of folds should be an integer."
        assert n_fold > 0, \
            "The number of folds should be positive: %d<0" % (n_fold)
        
        self.alpha = np.asarray(alpha)
        self.bandwidth = bandwidth
        self.optim_by_CV = optim_by_CV
        self.n_fold = n_fold

    @property
    def name(self):
        return self._name




# ---------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ---------------------------------------
#
# API for the quantile based QOSA indices 
#
# ---------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ---------------------------------------

class qosa_Quantile__Averaged_Quantile(DefaultParametersForForestMethod):
    """
    Computing the empirical conditional quantiles for each tree and then doing 
    the mean through the forest.

    Parameters
    ----------
    alpha : array-like of shape = [n_alphas]
        The order of the QOSA indices to compute.

    n_estimators : int, default=100
        The number of trees in the forest.

    min_samples_leaf : int or array-like of shape = [n_min_samples_leaf], default=20
        The minimum number of samples required to be at a leaf node. It is an 
        integer without CV, otherwise a ndarray if CV is used.
    
    used_bootstrap_samples: bool, default=False
        Using the bootstrap samples or the original sample to compute the 
        conditional quantiles.

    optim_by_CV : bool, default=False
        Cross validation method in order to find the best 'min_samples_leaf'
        hyperparameter for each variable in the model and each value of alpha.

    CV_strategy: str, default=None
        The estimation method used to do the cross validation.
        Available estimation methods are: "K_Fold" or "OOB"

    n_fold : int, default=3
        Number of folds for the cross validation method. Must be at least 2.
        This parameter is useful if you have selected "CV_strategy=K_Fold".

    random_state_Forest : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used when
        building trees and the sampling of the features to consider when looking
        for the best split at each node.
        
    References
    ----------
    .. [1] add reference
    """

    def __init__(self,
                 alpha,
                 n_estimators=100,
                 min_samples_leaf=20,
                 used_bootstrap_samples=False,
                 optim_by_CV=False,
                 CV_strategy=None,
                 n_fold=3,
                 random_state_Forest=None):

        super(qosa_Quantile__Averaged_Quantile, self).__init__(
                                            alpha=alpha,
                                            n_estimators=n_estimators,
                                            min_samples_leaf=min_samples_leaf,
                                            used_bootstrap_samples=used_bootstrap_samples,
                                            optim_by_CV=optim_by_CV,
                                            n_fold=n_fold,
                                            random_state_Forest=random_state_Forest)
        
        self._name = "Averaged_Quantile"
        self.CV_strategy = CV_strategy

    @DefaultParametersForForestMethod.CV_strategy.setter
    def CV_strategy(self, CV_strategy):
        if self.optim_by_CV == False and CV_strategy is not None:
            raise ValueError("CV_strategy needs to be None if you choose 'optim_by_CV=False'")
        if self.optim_by_CV == True and CV_strategy not in ("K_Fold", "OOB"):
            raise ValueError("Only 'K_Fold' or 'OOB' available for the argument"
                             " CV_strategy when 'optim_by_CV=True'")
        self._CV_strategy = CV_strategy


class qosa_Quantile__Weighted_CDF(DefaultParametersForForestMethod):
    """
    Compute the conditional quantiles with the weights based on the Original 
    or Bootstrap samples according the value of the parameter "used_bootstrap_samples".

    Parameters
    ----------
    alpha : array-like of shape = [n_alphas]
        The order of the QOSA indices to compute.

    n_estimators : int, default=100
        The number of trees in the forest.

    min_samples_leaf : int or array-like of shape = [n_min_samples_leaf], default=20
        The minimum number of samples required to be at a leaf node. It is an 
        integer without CV, otherwise a ndarray if CV is used.

    used_bootstrap_samples: bool, default=False
        Using the bootstrap samples or the original sample to compute the 
        conditional quantiles.

    optim_by_CV : bool, default=False
        Cross validation method in order to find the best 'min_samples_leaf'
        hyperparameter for each variable in the model and each value of alpha.

    CV_strategy: str, default=None
        The estimation method used to do the cross validation.
        Available estimation methods are: "K_Fold" or "OOB"

    n_fold : int, default=3
        Number of folds for the cross validation method. Must be at
        least 2.

    random_state_Forest : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used when
        building trees and the sampling of the features to consider when looking
        for the best split at each node.
        
    References
    ----------
    .. [1] add reference
    """

    def __init__(self,
                 alpha,
                 n_estimators=100,
                 min_samples_leaf=20,
                 used_bootstrap_samples=False,
                 optim_by_CV=False,
                 CV_strategy=None,
                 n_fold=3,
                 random_state_Forest=None):

        super(qosa_Quantile__Weighted_CDF, self).__init__(
                                        alpha=alpha,
                                        n_estimators=n_estimators,
                                        min_samples_leaf=min_samples_leaf,
                                        used_bootstrap_samples=used_bootstrap_samples,
                                        optim_by_CV=optim_by_CV,
                                        n_fold=n_fold,
                                        random_state_Forest=random_state_Forest)

        self._name = "Weighted_CDF"
        self.CV_strategy = CV_strategy

    @DefaultParametersForForestMethod.CV_strategy.setter
    def CV_strategy(self, CV_strategy):
        if self.optim_by_CV == False and CV_strategy is not None:
            raise ValueError("CV_strategy needs to be None if you choose 'optim_by_CV=False'")
        if self.optim_by_CV == True and CV_strategy not in ("K_Fold", "OOB"):
            raise ValueError("Only 'K_Fold' or 'OOB' available for the argument"
                             " CV_strategy when 'optim_by_CV=True'")
        self._CV_strategy = CV_strategy


class qosa_Quantile__Kernel_CDF(DefaultParametersForKernelMethod):
    """
    Estimate the conditional CDF thanks to the Kernel estimation.

    Parameters
    ----------
    alpha : array-like of shape = [n_alphas]
        The order of the QOSA indices to compute.

    bandwidth : float or array-like of shape = [n_bandwidth], default=None
        The bandwidth parameter used in the estimation method. It will be fixed 
        to n**(-1/5) in the algorithm if 'bandwidth=None'.

    optim_by_CV : bool, default=False
        Cross validation method in order to find the better 'bandwidth'
        parameter for each variable in the model and each value of alpha.

    n_fold : int, default=3
        Number of folds for the cross validation method. Must be at
        least 2.
    
    References
    ----------
    .. [1] Maume-Deschamps, Véronique, and Ibrahima Niang. "Estimation of 
           quantile oriented sensitivity indices." Statistics & Probability 
           Letters 134 (2018): 122-127.
    """

    def __init__(self,
                 alpha,
                 bandwidth=None,
                 optim_by_CV=False,
                 n_fold=3):

        super(qosa_Quantile__Kernel_CDF, self).__init__(
                                                    alpha=alpha,
                                                    bandwidth=bandwidth,
                                                    optim_by_CV=optim_by_CV,
                                                    n_fold=n_fold)

        self._name = "Kernel_CDF"




# --------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# --------------------------------------
#
# API for the minimum based QOSA indices 
#
# --------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# --------------------------------------

class qosa_Min__Min_in_Leaves(DefaultParametersForForestMethod):
    """
    Computing the expected value of the check function by an average of the
    minimum got within each leaf over all those of each tree.

    Parameters
    ----------
    alpha : array-like of shape = [n_alphas]
        The order of the QOSA indices to compute.

    n_estimators : int, default=100
        The number of trees in the forest.

    min_samples_leaf : int or array-like of shape = [n_min_samples_leaf], default=20
        The minimum number of samples required to be at a leaf node. It is an integer
        without CV, otherwise a ndarray if CV is used.

    used_bootstrap_samples: bool, default=False
        Using the bootstrap samples or the original sample to compute the minimum
        in each leaf.

    optim_by_CV : bool, default=False
        Cross validation method in order to find the best 'min_samples_leaf'
        hyperparameter for each variable in the model and each value of alpha.

    n_fold : int, default=3
        Number of folds for the cross validation method. Must be at
        least 2.

    random_state_Forest : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used when
        building trees and the sampling of the features to consider when looking
        for the best split at each node.

    References
    ----------
    .. [1] add reference
    """

    def __init__(self,
                 alpha,
                 n_estimators=100,
                 min_samples_leaf=20,
                 used_bootstrap_samples=False,
                 optim_by_CV=False,
                 n_fold=3,
                 random_state_Forest=None):

        super(qosa_Min__Min_in_Leaves, self).__init__(
                                            alpha=alpha,
                                            n_estimators=n_estimators,
                                            min_samples_leaf=min_samples_leaf,
                                            used_bootstrap_samples=used_bootstrap_samples,
                                            optim_by_CV=optim_by_CV,
                                            n_fold=n_fold,
                                            random_state_Forest=random_state_Forest)
        
        self._name = "Min_in_Leaves"


class qosa_Min__Weighted_Min(DefaultParametersForForestMethod):
    """
    Computing the expected value of the check function by getting the minimum
    with a weighted mean.

    Parameters
    ----------
    alpha : array-like of shape = [n_alphas]
        The order of the QOSA indices to compute.

    n_estimators : int, default=100
        The number of trees in the forest.

    min_samples_leaf : int or array-like of shape = [n_min_samples_leaf], default=20
        The minimum number of samples required to be at a leaf node. It is an integer
        without CV, otherwise a ndarray if CV is used.

    used_bootstrap_samples: bool, default=False
        Using the Bootstrap or Original Data based weights to compute the minimum
        with the weighted mean.

    optim_by_CV : bool, default=False
        Cross validation method in order to find the better 'min_samples_leaf'
        parameter for each variable in the model and each value of alpha.

    n_fold : int, default=3
        Number of folds for the cross validation method. Must be at
        least 2.

    random_state_Forest : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used when
        building trees and the sampling of the features to consider when looking
        for the best split at each node.

    References
    ----------
    .. [1] add reference
    """

    def __init__(self,
                 alpha,
                 n_estimators=100,
                 min_samples_leaf=20,
                 used_bootstrap_samples=False,
                 optim_by_CV=False,
                 n_fold=3,
                 random_state_Forest=None):

        super(qosa_Min__Weighted_Min, self).__init__(
                                        alpha=alpha,
                                        n_estimators=n_estimators,
                                        min_samples_leaf=min_samples_leaf,
                                        used_bootstrap_samples=used_bootstrap_samples,
                                        optim_by_CV=optim_by_CV,
                                        n_fold=n_fold,
                                        random_state_Forest=random_state_Forest)

        self._name = "Weighted_Min"


class qosa_Min__Weighted_Min_with_complete_forest(DefaultParametersForForestMethod):
    """
    Computing the expected value of the check function by getting the minimum
    with a weighted mean.

    Parameters
    ----------
    alpha : array-like of shape = [n_alphas]
        The order of the QOSA indices to compute.

    n_estimators : int, default=100
        The number of trees in the forest.

    min_samples_leaf : int or array-like of shape = [n_min_samples_leaf], default=20
        The minimum number of samples required to be at a leaf node. It is an integer
        without CV, otherwise a ndarray if CV is used.

    used_bootstrap_samples: bool, default=False
        Using the Bootstrap or Original Data based weights to compute the minimum
        with the weighted mean.

    optim_by_CV : bool, default=False
        Cross validation method in order to find the better 'min_samples_leaf'
        parameter for each variable in the model and each value of alpha.

    n_fold : int, default=3
        Number of folds for the cross validation method. Must be at
        least 2.

    random_state_Forest : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used when
        building trees and the sampling of the features to consider when looking
        for the best split at each node.
        
    References
    ----------
    .. [1] add reference
    """

    def __init__(self,
                 alpha,
                 n_estimators=100,
                 min_samples_leaf=20,
                 used_bootstrap_samples=False,
                 optim_by_CV=False,
                 n_fold=3,
                 random_state_Forest=None):

        super(qosa_Min__Weighted_Min_with_complete_forest, self).__init__(
                                        alpha=alpha,
                                        n_estimators=n_estimators,
                                        min_samples_leaf=min_samples_leaf,
                                        used_bootstrap_samples=used_bootstrap_samples,
                                        optim_by_CV=optim_by_CV,
                                        n_fold=n_fold,
                                        random_state_Forest=random_state_Forest)

        self._name = "Weighted_Min_with_complete_forest"


class qosa_Min__Kernel_Min(DefaultParametersForKernelMethod):
    """
    Estimate the QOSA indices by getting the min with a kernel method.

    Parameters
    ----------
    alpha : array-like of shape = [n_alphas]
        The order of the QOSA indices to compute.

    bandwidth : float or array-like of shape = [n_bandwidth], default=None
        The bandwidth parameter used in the estimation method. It will be fixed 
        to n**(-1/5) in the algorithm if 'bandwidth=None'.

    optim_by_CV : bool, default=False
        Cross validation method in order to find the better 'bandwidth'
        parameter for each variable in the model and each value of alpha.

    n_fold : int, default=3
        Number of folds for the cross validation method. Must be at
        least 2.

    References
    ----------
    .. [1] Browne, Thomas, et al. "Estimate of quantile-oriented sensitivity
           indices.", Preprint on HAL, 2017.
    """

    def __init__(self,
                 alpha,
                 bandwidth=None,
                 optim_by_CV=False,
                 n_fold=3):
        
        super(qosa_Min__Kernel_Min, self).__init__(alpha=alpha,
                                         bandwidth=bandwidth,
                                         optim_by_CV=optim_by_CV,
                                         n_fold=n_fold)

        self._name = "Kernel_Min"