# # -*- coding: utf-8 -*-

# """
# Add docstring of the module
# """


# import numba as nb
# import numpy as np
# from numpy import float64 as DTYPE
# from sklearn.utils import check_array


# __all__ = ['UnivariateQuantileRegressionKernel']


# class UnivariateQuantileRegressionKernel(object):
#     """
#     Univariate Quantile Regresion Kernel

#     This class computes the conditional quantiles thanks to the Kernel estimation
#     of the Cumulative Distributive Function.
#     """
    
#     def fit(self, X, y):
#         """     
#         Method allowing to provide and check the Monte-Carlo samples which will
#         be necessary to compute the conditional quantiles.

#         Parameters
#         ----------  
#         X : array-like of shape = [n_samples]
#             Input samples used to estimate the conditional CDF.
        
#         y : array-like of shape = [n_samples]
#             Output samples used to estimate the conditional CDF.

#         Returns
#         -------
#         self : object
#         """
        
#         # Transformation of the data in np.array
#         X = np.asarray(X)
#         y = np.asarray(y)

#         # Validate or convert the input data
#         X = check_array(X, ensure_2d=False, dtype=DTYPE)   
#         y = check_array(y, ensure_2d=False, dtype=DTYPE)   
        
#         if len(X.shape) == 2 and (X.shape[0] == 1 or X.shape[1] == 1):
#             X = X.ravel()
#         elif len(X.shape) == 2 and (X.shape[0] >= 2 and X.shape[1] >= 2):
#             raise ValueError("X dimension is not right. This method is done "
#                              "for univariate random variable.")
#         elif len(X.shape) >= 3:
#             raise ValueError("X dimension is not right. This method is done "
#                              "for univariate random variable.")
#         assert X.shape[0] == y.shape[0], "X and y do not have consistent length."
        
#         # Save the data. Necessary to compute the quantiles.
#         self._input_samples = X
#         self._output_samples = y
        
#         # Informations about the problem dimension
#         self._n_samples = X.shape[0]

#         return self

#     def predict(self, X, alpha, bandwidth=None):
#         """
#         Compute the conditional quantiles of order alpha of the elements X

#         Parameters
#         ----------
#         X : array-like of shape = [n_samples]
#             The elements where we want to assess the conditional quantiles.
        
#         alpha : array-like of shape = [n_alphas]
#             The order of the conditional quantiles to assess.

#         bandwidth : float, default = None
#             The bandwidth parameter used in the estimation of the conditional CDF.
#             With 'bandwidth=None' as argument, the value of the bandwidth in the
#             algorithm will be 'bandwidth=n**(-0.2)'

#         Returns
#         -------
#         quantiles : array-like of shape = [n_samples, n_alphas]
#             The conditional quantiles computed at points X for the different
#             values of alpha.
#         """
        
#         # Validate or convert the input data
#         if isinstance(X, (int, np.integer, float, np.floating)):
#             X = [X]
#         X = np.asarray(X)
#         X = check_array(X, ensure_2d=False, dtype=DTYPE)   
#         if len(X.shape) == 2 and (X.shape[0] == 1 or X.shape[1] == 1):
#             X = X.ravel()
#         elif len(X.shape) == 2 and (X.shape[0] >= 2 and X.shape[1] >= 2):
#             raise ValueError("X dimension is not right. This method is done "
#                              "for univariate random variable.")
#         elif len(X.shape) >= 3:
#             raise ValueError("X dimension is not right. This method is done "
#                              "for univariate random variable.")
        
#         if bandwidth is not None:
#             assert isinstance(bandwidth, (float, np.floating)), \
#                 "The 'bandwidth' parameter should be a float."
#             assert bandwidth>0, "The bandwidth should be positive: %f<0" % (bandwidth,)
#         else:
#             bandwidth = (self._n_samples)**(-0.2)

#         if isinstance(alpha, (int, np.integer, float, np.floating)):
#             alpha = [alpha]
#         alpha = np.asarray(alpha)

#         # Number of quantiles to compute
#         n_quantiles = X.shape[0]
#         n_alphas = alpha.size # Number of probabilities
        
#         quantiles = np.empty((n_quantiles, n_alphas), dtype=np.float64, order='C')
        
#         _compute_conditional_quantile_with_kernel_CDF(quantiles, 
#                                                       self._input_samples,
#                                                       self._output_samples,
#                                                       X,
#                                                       alpha,
#                                                       bandwidth)
#         return quantiles




# # ----------------------------------------------------
# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# # ----------------------------------------------------
# #
# # Private ancillary functions for the previous classes
# #
# # ----------------------------------------------------
# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# # ----------------------------------------------------

# # ------------------------------------------------
# # For the class UnivariateQuantileRegressionKernel
# # ------------------------------------------------

# @nb.njit("float64[:](float64[:])", nogil=True, cache=False, parallel=False)
# def _gaussian_kernel(x):
#     pi = np.sqrt(2*np.pi)
#     return np.exp(-0.5*x**2)/pi


# @nb.njit(nogil=True, cache=False, parallel=False)
# def _gaussian_kernel_2(x, input_samples_sorted, bandwidth, CDF):
#     pi = np.sqrt(2*np.pi)
#     for i in range(input_samples_sorted.shape[0]):
#         CDF[i] = np.exp(-0.5*((x-input_samples_sorted[i])/bandwidth)**2)/pi
        
        

# @nb.guvectorize([(nb.float64, nb.float64[:], nb.float64, nb.float64[:])], '(),(n),()->(n)')
# def _gaussian_kernel_3(x, input_samples_sorted, bandwidth, CDF):
#     pi = np.sqrt(2*np.pi)
#     for i in range(input_samples_sorted.shape[0]):
#         CDF[i] = np.exp(-0.5*((x-input_samples_sorted[i])/bandwidth)**2)/pi


# @nb.njit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:], float64)", nogil=True, cache=False, parallel=True)
# def _compute_conditional_quantile_with_kernel_CDF(quantiles, input_samples, output_samples, X, alpha, bandwidth):
    
#     n_quantiles = X.shape[0]
#     order_statistic = np.argsort(output_samples)
#     input_samples_sorted = input_samples[order_statistic]
    
#     for i in nb.prange(n_quantiles):
# #        CDF_temp = _gaussian_kernel_bis((X[i] - input_samples_sorted)/bandwidth)
        
#         CDF_temp = np.empty(input_samples_sorted.shape[0], dtype=np.float64)
#         _gaussian_kernel_3(X[i], input_samples_sorted, bandwidth, CDF_temp)
        
#         # Compute the quantiles thanks to the Cumulative Distribution Function 
#         # for each value of alpha
#         csum = np.cumsum(CDF_temp) 
#         CDF = csum / csum[-1]
    
#         quantiles[i, :] = np.array([
#                                 output_samples[
#                                     order_statistic[
#                                         np.argmax((CDF >= alpha_var).astype(np.uint32))]
#                                               ] 
#                                 for alpha_var in alpha])



# if __name__ == '__main__':
#     import time 
    
#     n_samples = 10**4
#     X = np.random.exponential(size = (n_samples,2))
#     y = X[:,0] - X[:,1]
#     X = X[:,0]
    
#     uqrk = UnivariateQuantileRegressionKernel()
#     uqrk.fit(X, y)
    
#     alpha = np.array([0.1, 0.5, 0.9])
#     X_value = np.random.exponential(size=(n_samples))
        
#     import cProfile
#     import pstats
    
    
#     start = time.time()    
# #    profiler = cProfile.Profile(builtins = False)
# #    profiler.enable()
#     a = uqrk.predict(X=X_value, alpha=alpha)
# #    profiler.disable()
# #    profiler.dump_stats('normal_profiler')
# #    pstats.Stats('normal_profiler').strip_dirs().sort_stats('time').print_stats(10)
#     print("Total time = ",time.time() - start)

# #    from qosa import cross_validation_kernel
# #    
# #    bandwidth = np.unique(np.logspace(np.log10(0.020), np.log10(0.50), 20, endpoint=True, dtype=np.float64))
# #    bandwidth = np.linspace(0.010, 0.50, 20, endpoint=True, dtype=np.float64)
# #    a, b = cross_validation_kernel(X=X, y=y, alpha=alpha, bandwidth=bandwidth)
# #    
#     from qosa import QuantileRegressionForest
    
# #    from qosa import cross_validation_forest
# #    min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(1500), 10, endpoint=True, dtype=int))
# #    cross_validation_forest(X=X[:,0], y=y, alpha=alpha, min_samples_leaf=min_samples_leaf, n_estimators=10, method="Averaged_Quantile")
    
#     qrf = QuantileRegressionForest(n_estimators=10, min_samples_leaf=60)
#     qrf.fit(X, y)
    
#     start = time.time()
#     qrf.predict(X_value, alpha)
#     print("Total time = ",time.time() - start)