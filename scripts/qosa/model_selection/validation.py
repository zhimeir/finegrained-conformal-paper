# -*- coding: utf-8 -*-

"""
Add docstring of the module
"""


import numpy as np
import pandas as pd
from numpy import float64 as DTYPE
from time import perf_counter
from sklearn.model_selection import KFold
from sklearn.utils import check_array, check_X_y

from ._validation import cross_validate
from ..base_forest import QuantileRegressionForest
from ..base_forest import _averaged_check_function_alpha_array
from ..base_kernel import UnivariateQuantileRegressionKernel


__all__ = ['cross_validation_forest', 'cross_validation_kernel']


def cross_validation_forest(X,
                            y,
                            alpha,
                            min_samples_leaf,
                            method="Weighted_CDF",
                            n_estimators=100,
                            used_bootstrap_samples=False,
                            CV_strategy="K_Fold",
                            n_fold=3,
                            shuffle=True):
    """
    Function to compute the averaged check function by cross-validation for
    several parameters for 'min_samples_leaf'.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        Input samples used to calibrate the forest.
        
    y : array-like, shape = [n_samples]
        Output samples used to calibrate the forest.

    alpha : array-like of shape = [n_alphas]
        The averaged check function computed for these alpha.

    min_samples_leaf : array-like of shape = [n_min_samples_leaf]
        The minimum number of samples required to be at a leaf node.
        The values to calculate the function with the cross validation strategy.

    method : str, default = "Weighted_CDF"
        Choose which method to use for computing the conditional quantiles among
        the following: "Averaged_Quantile" || "Weighted_CDF"

    n_estimators : int, optional (default=100)
        The number of trees in the forest.

    used_bootstrap_samples : bool, default=False
        Whether or not to use the bootstrap samples to compute the conditional
        quantiles.

    CV_strategy : str, default="K_Fold"
        The estimation method used to do the cross validation.
        Available estimation methods are "K_Fold" or "OOB" for the methods
        "Averaged_Quantile" or "Weighted_CDF".

    n_fold : int, optional (default=3)
        Number of folds for the cross validation method. Must be at least 2.

    shuffle : bool, optional (default=True)
        Whether to shuffle the data before splitting into batches.

    Returns
    -------
    df_cross_val_values : dataframe
    """

    # Transformation of the data in np.array
    X = np.asarray(X, order='C')
    y = np.asarray(y)

    # Validate or convert the input data
    try:
        X, y = check_X_y(X, y, dtype=DTYPE, y_numeric=True)
    except ValueError as instance_ValueError:
        if X.shape[0] == X.size:
            X = X.reshape(-1,1, order='C')
            X, y = check_X_y(X, y, dtype=DTYPE, y_numeric=True)
        else:
            raise ValueError(instance_ValueError)

    if isinstance(alpha, (int, np.integer, float)):
        alpha = [alpha]
    assert isinstance(min_samples_leaf, np.ndarray), \
        "The minimum number of samples required to be a leaf node should be an array."
    assert isinstance(method, str), \
        "The parameter 'method' should be a string. Given %s" % (type(method),)
    assert method in ("Averaged_Quantile", "Weighted_CDF"), \
        "Given method not allowed: %s" % (method,)
    assert isinstance(n_estimators, (int, np.integer)), \
        "The number of tree should be an integer."
    assert n_estimators > 0, \
        "The number of tree should be positive: %d<0" % (n_estimators)
    assert isinstance(used_bootstrap_samples, bool), \
            "The parameter 'used_bootstrap_samples' should be a boolean."
    assert isinstance(CV_strategy, str), \
        "The parameter 'CV_strategy' should be a string. Given %s" % (type(method),)
    assert isinstance(n_fold, (int, np.integer)), "The number of folds should be an integer."
    assert n_fold > 0, "The number of folds should be positive: %d<0" % (n_fold)
    assert isinstance(shuffle, bool), \
        "The parameter to shuffle the data should be a boolean."

    alpha = np.asarray(alpha)
    n_alphas = alpha.shape[0]
    
    if method in ("Averaged_Quantile", "Weighted_CDF") and CV_strategy == "K_Fold":
        def check_function_error(estimator, X_test, y_test, alpha, method, used_bootstrap_samples):
            conditional_quantiles = estimator.predict_quantile(
                                           X=X_test, 
                                           alpha=alpha,
                                           method=method,
                                           used_bootstrap_samples=used_bootstrap_samples)
            u = y_test.reshape(-1,1) - conditional_quantiles
            return _averaged_check_function_alpha_array(u, alpha)

        n_leaf = min_samples_leaf.shape[0]    
        cross_val_values = np.empty((n_leaf, n_fold, n_alphas), dtype=np.float64)
        
        seed_cv = int(perf_counter())
        seed_estimator = int((perf_counter()+1000)/10)
        cv = KFold(n_splits=n_fold, shuffle=shuffle, random_state=seed_cv)
        scoring_params = {'alpha':alpha, 
                          'method':method,
                          'used_bootstrap_samples':used_bootstrap_samples}

        for i in range(n_leaf):
            cross_val_values[i, :, :] = cross_validate(
                                        estimator=QuantileRegressionForest(
                                                            n_estimators=n_estimators,
                                                            min_samples_split=min_samples_leaf[i]*2, 
                                                            min_samples_leaf=min_samples_leaf[i],
                                                            random_state=seed_estimator), 
                                        X=X,
                                        y=y,
                                        cv=cv,
                                        scoring=check_function_error,
                                        scoring_params=scoring_params)

    elif method in ("Averaged_Quantile", "Weighted_CDF") and CV_strategy == "OOB":
        n_leaf = min_samples_leaf.shape[0]  
        cross_val_values = np.empty((n_leaf, 1, n_alphas), dtype=np.float64)
        seed_estimator = int((perf_counter()+1000)/10)

        for i in range(n_leaf):
            qosa = QuantileRegressionForest(
                                        n_estimators=n_estimators,
                                        min_samples_split=min_samples_leaf[i]*2, 
                                        min_samples_leaf=min_samples_leaf[i],
                                        random_state=seed_estimator)
            qosa.fit(X, y, 
                     oob_score_quantile=True,
                     alpha=alpha,
                     method=method,
                     used_bootstrap_samples=used_bootstrap_samples)
            cross_val_values[i, :, :] = qosa.oob_score_quantile_

    # Formatting the results
    index_columns = ['$Fold_{%d}$' %(i+1) for i in range(n_fold)]
    index_columns.append('Mean_Fold')
    index_rows = [alpha, min_samples_leaf]
    index_rows = pd.MultiIndex.from_product(index_rows, names = ['Alpha', 'Min_samples_leaf'])
    df_cross_val_values = pd.DataFrame(index = index_rows, columns = index_columns)

    for i, alpha_temp in enumerate(alpha):
        df_cross_val_values.loc[alpha_temp].loc[:,['$Fold_{%d}$' %(i+1) for i in range(n_fold)]] = cross_val_values[:,:,i]
        df_cross_val_values['Mean_Fold'].loc[alpha_temp][:] = cross_val_values[:,:,i].mean(axis=1)

    index_optim_min_samples_leaf = np.argmin(
                            df_cross_val_values['Mean_Fold'].reset_index(
                                        inplace=False).pivot(
                                                    index='Min_samples_leaf',
                                                    columns='Alpha',
                                                    values='Mean_Fold').values,
                            axis=0)

    return df_cross_val_values, min_samples_leaf[index_optim_min_samples_leaf]




def cross_validation_kernel(X,
                            y,
                            alpha,
                            bandwidth,
                            n_fold=3,
                            shuffle=True):
    """
    Function to compute the averaged check function by cross-validation for
    several parameters for 'bandwidth'.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        Input samples.
        
    y : array-like, shape = [n_samples]
        Output samples.

    alpha : array-like of shape = [n_alphas]
        The averaged check function computed for these alpha levels.

    bandwidth : array-like of shape = [n_bandwidth]
        The bandwidth parameter used in the estimation of the conditional CDF.
        The values to assess the function with the cross validation strategy.

    n_fold : int, optional (default=3)
        Number of folds for the cross validation method. Must be at least 2.

    shuffle : bool, optional (default=True)
        Whether to shuffle the data before splitting into batches.

    Returns
    -------
    df_cross_val_values : dataframe
    """

    # Transformation of the data in np.array
    X = np.asarray(X)
    y = np.asarray(y)

    # Validate or convert the input data
    X = check_array(X, ensure_2d=False, dtype=DTYPE)   
    y = check_array(y, ensure_2d=False, dtype=DTYPE)   
    
    if len(X.shape) == 2 and (X.shape[0] == 1 or X.shape[1] == 1):
        X = X.ravel()
    elif len(X.shape) == 2 and (X.shape[0] >= 2 and X.shape[1] >= 2):
        raise ValueError("X dimension is not right. This method is done "
                         "for univariate random variable.")
    elif len(X.shape) >= 3:
        raise ValueError("X dimension is not right. This method is done "
                         "for univariate random variable.")
    assert X.shape[0] == y.shape[0], "X and y do not have consistent length."

    if isinstance(alpha, (int, np.integer, float, np.floating)):
        alpha = [alpha]
    assert isinstance(bandwidth, np.ndarray), \
        "The bandwidth parameter to estimate the conditional CDF should be an array."
    assert isinstance(n_fold, (int, np.integer)), "The number of folds should be an integer."
    assert n_fold > 0, "The number of folds should be positive: %d<0" % (n_fold)
    assert isinstance(shuffle, bool), \
        "The parameter to shuffle the data should be a boolean."

    alpha = np.asarray(alpha)
    n_alphas = alpha.shape[0]
    
    def check_function_error(estimator, X_test, y_test, alpha, bandwidth):
        conditional_quantiles = estimator.predict(X=X_test, 
                                                  alpha=alpha,
                                                  bandwidth=bandwidth)
        u = y_test.reshape(-1,1) - conditional_quantiles
        return _averaged_check_function_alpha_array(u, alpha)

    n_bandwidth = bandwidth.shape[0] 
    cross_val_values = np.empty((n_bandwidth, n_fold, n_alphas), dtype=np.float64)
    
    seed_cv = int(perf_counter())
    cv = KFold(n_splits=n_fold, shuffle=shuffle, random_state=seed_cv)

    for i in range(n_bandwidth):
        scoring_params = {'alpha':alpha, 
                          'bandwidth':bandwidth[i]}

        cross_val_values[i, :, :] = cross_validate(
                                        estimator=UnivariateQuantileRegressionKernel(), 
                                        X=X,
                                        y=y,
                                        cv=cv,
                                        scoring=check_function_error,
                                        scoring_params=scoring_params)

    # Formatting the results
    index_columns = ['$Fold_{%d}$' %(i+1) for i in range(n_fold)]
    index_columns.append('Mean_Fold')
    index_rows = [alpha, bandwidth]
    index_rows = pd.MultiIndex.from_product(index_rows, names = ['Alpha', 'Bandwidth'])
    df_cross_val_values = pd.DataFrame(index = index_rows, columns = index_columns)

    for i, alpha_temp in enumerate(alpha):
        df_cross_val_values.loc[alpha_temp].loc[:,['$Fold_{%d}$' %(i+1) for i in range(n_fold)]] = cross_val_values[:,:,i]
        df_cross_val_values['Mean_Fold'].loc[alpha_temp][:] = cross_val_values[:,:,i].mean(axis=1)

    index_optim_bandwidth = np.argmin(
                                df_cross_val_values['Mean_Fold'].reset_index(
                                            inplace=False).pivot(
                                                        index='Bandwidth',
                                                        columns='Alpha',
                                                        values='Mean_Fold').values,
                                axis=0)

    return df_cross_val_values, bandwidth[index_optim_bandwidth]