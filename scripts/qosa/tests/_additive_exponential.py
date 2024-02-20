# -*- coding: utf-8 -*-

"""
Module allowing to compute the true Qosa indices for a Hypo-exponential 
distribution as model's output
"""


import numpy as np
from scipy.optimize import fsolve, root_scalar


__all__ = ['qhypoexp', 'compute_qosa_indices_additive_exponential']


def qhypoexp(alpha, rates, x0=20, xtol=10**(-8), maxfev=10**9, factor=2):
    if isinstance(alpha, (int, np.integer, float)):
        alpha = [alpha]
    alpha = np.asarray(alpha)
    
    n_alphas = alpha.size
    res = np.zeros(n_alphas)
    for i, alpha_temp in enumerate(alpha):
        func = lambda x: _quantile_objective(x, alpha_temp, rates) 
        # res[i] = fsolve(func, x0=x0, xtol=xtol, maxfev=maxfev, factor=factor)
        res[i] = root_scalar(func, bracket=[0.0, 1.0e+10], x0=x0, xtol=xtol, maxiter=maxfev).root
        # Pour le x0 : estimation initiale
        # bracket=[0.0, 1.0e+10] vient du package sdprisk de R dans la fonction qhypoexp
        # xtol : tolérance pr savoir quand se finit le calcul
        # maxfev : nbre maximum d'appels à la fonction
        # factor : un paramètre déterminant le saut de l'état initial 0.1<factor<100 !!!!!!!! Paramèrtre très important
    return res


def _quantile_objective(x, alpha, rates):
    n_rates = rates.size
    var = 0
    
    for i in range(n_rates):
        rates_minus_i = np.delete(rates, i)
        polynom_i = np.prod(rates_minus_i/(rates_minus_i-rates[i]))
        var += np.exp(-rates[i]*x)*polynom_i
    return (1 - var) - alpha


def truncated_expectation(rates, d):
    n_rates = rates.size
    t1 = -d*np.exp(-rates*d) + (1-np.exp(-rates*d))/rates
    t2 = np.zeros(n_rates)

    for i in range(n_rates): 
        rates_minus_i = np.delete(rates, i)
        t2[i] = np.prod(rates_minus_i/(rates_minus_i-rates[i]))
    
    res = (t1*t2).sum()
    return res

def compute_qosa_indices_additive_exponential(rates, alpha):
    n_alphas = alpha.size
    n_rates = rates.size

    mean = (1/rates).sum()
    alpha_quantile = qhypoexp(alpha, rates)
    
    denominator = np.zeros(n_alphas)
    for i, alpha_temp in enumerate(alpha):
        denominator[i] = (alpha_temp*mean - 
                          truncated_expectation(rates, alpha_quantile[i]))

    numerator = np.zeros((n_rates, n_alphas))
    for i in range(n_rates):
        rates_minus_i = np.delete(rates, i)
        conditional_quantile = qhypoexp(alpha, rates_minus_i)
        for j in range(n_alphas):
            cond = truncated_expectation(rates_minus_i, conditional_quantile[j])
            numerator[i,j] = alpha[j]*(1/rates_minus_i).sum() - cond

    qosa_indices = 1 - numerator/denominator

    return qosa_indices.T
