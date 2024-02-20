# -*- coding: utf-8 -*-

"""
Add docstring of the module
"""


import numba as nb
import numpy as np
from numpy import float64 as DTYPE
from sklearn.utils import check_array


__all__ = ['UnivariateQuantileRegressionKernel']


class BaseEstimatorKernel(object):
    """
    Base class for all estimators using a Kernel method in their operation.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    
    def fit(self, X, y):
        """     
        Method allowing to provide and check the Monte-Carlo samples which will
        be necessary to use the Kernel methods.

        Parameters
        ----------  
        X : array-like of shape = [n_samples]
            Input samples.
        
        y : array-like of shape = [n_samples]
            Output samples.

        Returns
        -------
        self : object
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
        
        # Save the data.
        self._input_samples = X
        self._output_samples = y
        
        # Informations about the problem dimension
        self._n_samples = X.shape[0]

        return self    


class UnivariateQuantileRegressionKernel(BaseEstimatorKernel):
    """
    Univariate Quantile Regresion Kernel

    This class computes the conditional quantiles thanks to the Kernel based 
    estimation of the conditional Cumulative Distributive Function.
    """

    def predict(self, X, alpha, bandwidth=None):
        """
        Compute the conditional quantiles at the alpha level for the elements X.

        Parameters
        ----------
        X : array-like of shape = [n_samples]
            The elements where we want to assess the conditional quantiles.
        
        alpha : array-like of shape = [n_alphas]
            The level of the conditional quantiles to assess.

        bandwidth : float, default = None
            The bandwidth parameter used in the estimation of the conditional CDF.
            With 'bandwidth=None' as argument, the value of the bandwidth in the
            algorithm will be 'bandwidth=n**(-0.2)'

        Returns
        -------
        quantiles : array-like of shape = [n_samples, n_alphas]
            The conditional quantiles computed at points X for the different
            values of alpha. 
            If the alpha levels are provided in no particular order,
            e.g. alpha=[0.5, 0.1, 0.9], the returned values for each point / row 
            are sorted according to alpha levels, i.e. [0.1, 0.5, 0.9].
        """
        
        # Validate or convert the input data
        if isinstance(X, (int, np.integer, float, np.floating)):
            X = [X]
        X = np.asarray(X).astype(dtype=np.float64)
        X = check_array(X, ensure_2d=False, dtype=DTYPE)   
        if len(X.shape) == 2 and (X.shape[0] == 1 or X.shape[1] == 1):
            X = X.ravel()
        elif len(X.shape) == 2 and (X.shape[0] >= 2 and X.shape[1] >= 2):
            raise ValueError("X dimension is not right. This method is done "
                             "for univariate random variable.")
        elif len(X.shape) >= 3:
            raise ValueError("X dimension is not right. This method is done "
                             "for univariate random variable.")
        
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            alpha = [alpha]
        alpha = np.asarray(alpha).astype(dtype=np.float64)
        alpha.sort()

        if bandwidth is not None:
            assert bandwidth>0, "The bandwidth should be positive: %f<0" % (bandwidth,)
            bandwidth = np.float64(bandwidth)
        else:
            bandwidth = (self._n_samples)**(-0.2)

        # Number of quantiles to compute
        n_quantiles = X.shape[0]
        n_alphas = alpha.size # Number of probabilities
        
        quantiles = np.empty((n_quantiles, n_alphas), dtype=np.float64, order='C')
        
        _compute_conditional_quantile_with_kernel_CDF(quantiles, 
                                                      self._input_samples,
                                                      self._output_samples,
                                                      X,
                                                      alpha,
                                                      bandwidth)
        return quantiles


class MinimumConditionalExpectedCheckFunctionKernel(BaseEstimatorKernel):
    """
    Minimum Conditional Expected Check Function Kernel

    This class computes the expected value of the check function used in the 
    QOSA index. The idea is to look for the minimum of a kernel estimate of 
    the conditional check function.
    """

    def predict(self, X, X_density, alpha, bandwidth=None):
        """
        Compute the expected optimal value of the conditional contrast function
        for the elements X at the alpha order. A detailed description can be 
        found in [1].
        
        Parameters
        ----------
        X : array-like of shape = [n_samples]
            The elements where we want to assess the conditional contrast function.

        X_density : array-like of shape = [n_samples]
            The pdf computed for each point contained in X.
        
        alpha : array-like of shape = [n_alphas]
            The alpha order in which to evaluate the conditional contrast function.

        bandwidth : float, default = None
            The bandwidth parameter used in the estimation method.
            With 'bandwidth=None' as argument, the value of the bandwidth in the
            algorithm will be 'bandwidth=n**(-0.2)'

        Returns
        -------
        minimum_expectation : array-like of shape = [n_samples, n_alphas]
            The expected value of the check function computed at points X for
            the different values of alpha.
            If the alpha levels are provided in no particular order,
            e.g. alpha=[0.5, 0.1, 0.9], the returned values for each point / row 
            are sorted according to alpha levels, i.e. [0.1, 0.5, 0.9].
            
        References
        ----------
        .. [1] Browne, Thomas, et al. "Estimate of quantile-oriented sensitivity
               indices.", Preprint on HAL, 2017.
        """
        
        # Validate or convert the input data
        if isinstance(X, (int, np.integer, float, np.floating)):
            X = [X]
        X = np.asarray(X).astype(dtype=np.float64)
        X = check_array(X, ensure_2d=False, dtype=DTYPE)   
        if len(X.shape) == 2 and (X.shape[0] == 1 or X.shape[1] == 1):
            X = X.ravel()
        elif len(X.shape) == 2 and (X.shape[0] >= 2 and X.shape[1] >= 2):
            raise ValueError("X dimension is not right. This method is done "
                             "for univariate random variable.")
        elif len(X.shape) >= 3:
            raise ValueError("X dimension is not right. This method is done "
                             "for univariate random variable.")

        if isinstance(X_density, (int, np.integer, float, np.floating)):
            X_density = [X_density]
        X_density = np.asarray(X_density).astype(dtype=np.float64)
        assert X.shape[0] == X_density.shape[0], \
            "X and X_density do not have consistent length."
        
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            alpha = [alpha]
        alpha = np.asarray(alpha).astype(dtype=np.float64)
        alpha.sort() # Very important to have alpha sorted for the numba function.

        if bandwidth is not None:
            assert bandwidth>0, "The bandwidth should be positive: %f<0" % (bandwidth,)
            bandwidth = np.float64(bandwidth)
        else:
            bandwidth = (self._n_samples)**(-0.2)

        # Number of quantiles to compute
        n_minimum_expectation = X.shape[0]
        n_alphas = alpha.size # Number of probabilities
        
        minimum_expectation = np.empty((n_minimum_expectation, n_alphas), 
                                       dtype=np.float64,
                                       order='C')

        _compute_second_term_Browne_estimate(minimum_expectation, 
                                             self._input_samples,
                                             self._output_samples,
                                             X,
                                             X_density,
                                             alpha,
                                             bandwidth)
        
        return minimum_expectation




# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------
#
# Private ancillary functions for the previous classes
#
# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------

# -------------------------
# Function for both classes
# -------------------------

@nb.njit("void(float64[:], float64, float64[:], float64)", nogil=True, cache=True, parallel=False)
def _gaussian_kernel(kernel_evaluation, x, input_samples, bandwidth):

    pi = np.sqrt(2*np.pi)
    for i in range(input_samples.shape[0]):
        kernel_evaluation[i] = np.exp(-0.5*((x - input_samples[i])/bandwidth)**2)/pi


# ------------------------------------------------
# For the class UnivariateQuantileRegressionKernel
# ------------------------------------------------

@nb.njit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:], float64)", nogil=True, cache=True, parallel=True)
def _compute_conditional_quantile_with_kernel_CDF(quantiles, input_samples, output_samples, X, alpha, bandwidth):
    
    n_samples = input_samples.shape[0]
    n_quantiles = X.shape[0]
    order_statistic = np.argsort(output_samples)
    input_samples_sorted = input_samples[order_statistic]
    
    for i in nb.prange(n_quantiles):
        CDF_temp = np.empty(n_samples, dtype=np.float64)
        _gaussian_kernel(CDF_temp, X[i], input_samples_sorted, bandwidth)
        
        # Compute the quantiles thanks to the Conditional Cumulative Distribution 
        # Function for each value of alpha
        csum = np.cumsum(CDF_temp) 
        CDF = csum / csum[-1]

        quantiles[i, :] = np.array([
                                output_samples[
                                    order_statistic[
                                        np.argmax((CDF >= alpha_var).astype(np.uint32))]
                                              ] 
                                for alpha_var in alpha])


# -----------------------------------------------------------
# For the class MinimumConditionalExpectedCheckFunctionKernel
# -----------------------------------------------------------

@nb.njit("float64(float64[:], float64[:], float64, float64, float64, float64)", nogil=True, cache=True, parallel=False)
def _averaged_check_function_alpha_float_unparallel_product_kernel(output_samples, kernel, X_density_i, bandwidth, theta, alpha):

    n_samples = output_samples.shape[0]
    res = 0.    
    for i in range(n_samples):
        res += (output_samples[i] - theta)*(alpha - ((output_samples[i] - theta) < 0.))*kernel[i]
    
    res /= (n_samples*X_density_i*bandwidth)
    return res


@nb.njit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:], float64[:], float64)", nogil=True, cache=True, parallel=True)
def _compute_second_term_Browne_estimate(minimum_expectation, input_samples, output_samples, X, X_density, alpha, bandwidth):
    
    n_samples = input_samples.shape[0]
    n_minimum_expectation = X.shape[0]
    argsort_output_samples = np.argsort(output_samples)
    
    for i in nb.prange(n_minimum_expectation):    
        kernel = np.empty(n_samples, dtype=np.float64)
        _gaussian_kernel(kernel, X[i], input_samples, bandwidth)

        k=0
        for j, alpha_temp in enumerate(alpha):
            if k < n_samples:
                ans = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                     output_samples,
                                                     kernel,
                                                     X_density[i],
                                                     bandwidth,
                                                     output_samples[argsort_output_samples[k]],
                                                     alpha_temp)
            else:
                for l in range(j, alpha.shape[0]):
                    minimum_expectation[i, l] = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                     output_samples,
                                                     kernel,
                                                     X_density[i],
                                                     bandwidth,
                                                     output_samples[argsort_output_samples[-1]],
                                                     alpha[l])
                break

            if j == 0:
                k=1
            else:
                k+=1
            while(k < n_samples):
                temp = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                  output_samples,
                                                  kernel,
                                                  X_density[i],
                                                  bandwidth,
                                                  output_samples[argsort_output_samples[k]],
                                                  alpha_temp)
                if(temp <= ans):
                    ans = temp
                    k+=1
                else:
                    k-=1
                    break
            minimum_expectation[i, j] = ans