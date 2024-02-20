# -*- coding: utf-8 -*-

"""
Module allowing to compute the true Qosa indices for the toy insurance model
"""


import openturns as ot
import numba as nb
import numpy as np


__all__ = ['compute_qosa_indices_toy_insurance']


def compute_qosa_indices_toy_insurance(GPD_params, LN_params, Gamma_params, alpha, n_samples):   
    dim = 3
    margins = [ot.GeneralizedPareto(*GPD_params), ot.LogNormal(*LN_params), ot.Gamma(*Gamma_params)]
    copula = ot.IndependentCopula(dim)
    input_distribution = ot.ComposedDistribution(margins, copula)
    
    input_samples = np.array(input_distribution.getSample(n_samples), order='C').T
    EY = GPD_params[0]/(1 - GPD_params[1]) + np.exp(LN_params[0] + 0.5*LN_params[1]**2) + Gamma_params[0]/Gamma_params[1]
    
    qosa =  _compute_qosa_indices(input_samples, alpha, EY)
    
    return qosa.T


@nb.njit("float64[:,:](float64[:,:], float64[:], float64)", nogil=True, cache=True, parallel=True)
def _compute_qosa_indices(input_samples, alpha, esperance_Y):
    dim = input_samples.shape[0]
    n_alphas = alpha.shape[0]
    
    idx_variable = np.arange(dim)
    Y = input_samples.sum(axis=0)
    quantile_Y = np.percentile(Y, alpha*100)

    EY_truncated = np.empty((n_alphas), dtype=np.float64)
    for i in nb.prange(n_alphas):
        EY_truncated[i] = (Y*(Y <= quantile_Y[i])).mean()
        
    q_alpha_cond_variable = np.empty((dim, n_alphas), dtype=np.float64)           
    EY_truncated_variable = np.empty((dim, n_alphas), dtype=np.float64)
    for i in nb.prange(dim):
        idx_variable_temp = np.delete(idx_variable, i)
        input_samples_temp = input_samples[idx_variable_temp].sum(axis=0)
        q_alpha_cond_variable[i,:] = np.percentile(input_samples_temp, alpha*100)
        for j in range(n_alphas):
            EY_truncated_variable[i,j] = (Y*(input_samples_temp <= q_alpha_cond_variable[i,j])).mean()
    
    qosa = np.empty((dim, n_alphas), dtype=np.float64)
    for i in nb.prange(dim):
        qosa[i,:] = (EY_truncated_variable[i,:] - EY_truncated)/(alpha*esperance_Y - EY_truncated)
    
    return qosa