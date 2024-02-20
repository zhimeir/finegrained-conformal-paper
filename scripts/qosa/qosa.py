# -*- coding: utf-8 -*-

"""
Add docstring of the module
"""


import itertools
import numba as nb
import numpy as np
from multiprocessing import Pool
from numpy import float64 as DTYPE
from sklearn.utils import check_X_y, check_array

from .base_forest import (BaseEstimatorForest,
                          MinimumConditionalExpectedCheckFunctionWithLeaves,
                          MinimumConditionalExpectedCheckFunctionWithWeights,
                          QuantileRegressionForest,
                          _averaged_check_function_alpha_array,
                          _averaged_check_function_alpha_float_parallel,
                          _averaged_check_function_alpha_float_unparallel,
                          _averaged_check_function_alpha_float_unparallel_product_weight,
                          _compute_weight_with_bootstrap_data,
                          _compute_weight_with_original_data)
from .base_kernel import (MinimumConditionalExpectedCheckFunctionKernel,
                          UnivariateQuantileRegressionKernel)
from .indices import SensitivityResults
from .methods_for_qosa import *
from .model import ProbabilisticModel
from .model_selection import cross_validation_forest, cross_validation_kernel


__all__ = ['MinimumBasedQosaIndices', 'QuantileBasedQosaIndices']


class QuantileBasedQosaIndices(object):
    """
    The Quantile based Qosa indices.

    Estimate with a Monte-Carlo sampling the Quantile based Qosa indices. 
    Method developped by [1] as well as the new ones using the Random
    Forest [2] are implemented.
    
    References
    ----------
    .. [1] Maume-Deschamps, Véronique, and Ibrahima Niang. "Estimation of 
           quantile oriented sensitivity indices." Statistics & Probability 
           Letters 134 (2018): 122-127.
    .. [2] add reference
    """

    def build_sample(self, model, n_samples):
        """
        Creates the Monte-Carlo samples for the variables.

        This method creates the input samples and evaluate them through the 
        model to create the output samples.

        Parameters
        ----------
        model : ProbabilisticModel's instance.
            An object containing all the necessary information about the model.

        n_samples : int
            The sampling size for the Monte-Carlo estimation.
        """

        assert isinstance(model, ProbabilisticModel), \
            "The model should be an instance of the class ProbabilisticModel. Given %s" %(type(model),)
        assert isinstance(n_samples, (int, np.integer)), \
            "The number of sample should be an integer."
        assert n_samples > 0, \
            "The number of sample should be positive: %d<0" % (n_samples)

        input_distribution = model.input_distribution

        # Simulate the samples
        self.input_samples_1 = np.asarray(input_distribution.getSample(n_samples),
                                          order='F')
        self.input_samples_2 = np.asarray(input_distribution.getSample(n_samples),
                                          order='F')
        self.output_samples_1 = model(self.input_samples_1)
        self.output_samples_2 = model(self.input_samples_2)
        
        self._n_samples = n_samples
        self._dim = model.dim
        self._model = model

    def feed_sample(self, X1, Y1, X2, Y2):
        """
        The user provides the Monte-Carlo samples to compute the Qosa indices.

        Parameters
        ----------
        X1, X2 : array-like of shape = [n_samples, n_features]
            Input samples used to compute the QOSA indices
        
        Y1, Y2 : array-like, shape = [n_samples]
            Output samples used to compute the QOSA indices
        """

        # Transformation of the data in np.array
        X1 = np.asarray(X1, order='F')
        Y1 = np.asarray(Y1)
        X2 = np.asarray(X2, order='F')
        Y2 = np.asarray(Y2)

        # Validate or convert the input data
        try:
            X1, Y1 = check_X_y(X1, Y1, dtype=DTYPE, y_numeric=True)
        except ValueError as instance_ValueError:
            if X1.shape[0] == X1.size:
                X1 = X1.reshape(-1, 1, order='F')
                X1, Y1 = check_X_y(X1, Y1, dtype=DTYPE, y_numeric=True)
            else:
                raise ValueError(instance_ValueError)

        try:
            X2, Y2 = check_X_y(X2, Y2, dtype=DTYPE, y_numeric=True)
        except ValueError as instance_ValueError:
            if X2.shape[0] == X2.size:
                X2 = X2.reshape(-1, 1, order='F')
                X2, Y2 = check_X_y(X2, Y2, dtype=DTYPE, y_numeric=True)
            else:
                raise ValueError(instance_ValueError)

        assert X1.shape[1] == X2.shape[1], \
            "X1 and X2 do not have the same number of features."           

        self.input_samples_1 = X1
        self.input_samples_2 = X2
        self.output_samples_1 = Y1
        self.output_samples_2 = Y2
        
        # Informations about the problem dimension
        self._n_samples, self._dim = X1.shape
        self._model = None

    def compute_indices(self, method):
        """
        Method to compute the QOSA indices from the previously created samples
        thanks to 'build_sample' or 'feed_sample'.

        Parameters
        ----------
        method : object
            Choose which method to use for computing the conditional quantiles among
            the classes following: 
            - the forest methods: 'qosa_Quantile__Averaged_Quantile', 'qosa_Quantile__Weighted_CDF'      
            - the kernel method : 'qosa_Quantile__Kernel_CDF'

        Returns
        -------
        results : SensitivityResults instance
            The computed Qosa indices.
            If the alpha levels are provided in no particular order,
            e.g. alpha=[0.5, 0.1, 0.9], the returned values for each variable 
            are sorted according to alpha levels, i.e. [0.1, 0.5, 0.9].
        """

        assert isinstance(method, (qosa_Quantile__Averaged_Quantile, qosa_Quantile__Kernel_CDF,
                                   qosa_Quantile__Weighted_CDF)), "The provided method is not known. Given %s" % (type(method), )

        alpha = np.asarray(method.alpha).astype(dtype=np.float64)
        alpha.sort()
        dim = self._dim

        X1 = self.input_samples_1
        X2 = self.input_samples_2
        Y1 = self.output_samples_1
        Y2 = self.output_samples_2

        if method.name in _QUANTILE_ESTIMATORS_FOREST:
            results_indice_func = _qosa_forest_compute_quantile(
                                                    X1=X1,
                                                    X2=X2,
                                                    Y1=Y1,
                                                    Y2=Y2,
                                                    dim=dim,
                                                    alpha=alpha,
                                                    method=method.name,
                                                    n_estimators=method.n_estimators,
                                                    min_samples_leaf=method.min_samples_leaf,
                                                    used_bootstrap_samples=method.used_bootstrap_samples,
                                                    optim_by_CV=method.optim_by_CV,
                                                    CV_strategy=method.CV_strategy,
                                                    n_fold=method.n_fold,
                                                    random_state_Forest=method.random_state_Forest)
        elif method.name in _QUANTILE_ESTIMATORS_KERNEL:
            results_indice_func = _qosa_kernel_compute_quantile(
                                                    X1=X1,
                                                    X2=X2,
                                                    Y1=Y1,
                                                    Y2=Y2,
                                                    dim=dim,
                                                    alpha=alpha,
                                                    bandwidth=method.bandwidth,
                                                    optim_by_CV=method.optim_by_CV,
                                                    n_fold=method.n_fold)

        qosa_indices = results_indice_func[0]
        optimal_parameter_by_CV = results_indice_func[1]

        if isinstance(self._model, ProbabilisticModel):
            self._model.alpha = alpha
            true_qosa_indices = self._model.qosa_indices
        else:
            true_qosa_indices = None

        results = SensitivityResults(
                             alpha=alpha,
                             qosa_indices_estimates=qosa_indices,
                             true_qosa_indices=true_qosa_indices,
                             dim=dim,
                             optim_by_CV=method.optim_by_CV,
                             optimal_parameter_by_CV=optimal_parameter_by_CV,
                             method=method.name)

        return results


class MinimumBasedQosaIndices(object):
    """
    The Minimum based Qosa indices.

    Estimate with a Monte-Carlo sampling the minimum based qosa indices. 
    Method developped by [1]  as well as the new ones using the
    Random Forest [2] are implemented.
    
    References
    ----------
    .. [1] Browne, Thomas, et al. "Estimate of quantile-oriented sensitivity
           indices.", Preprint on HAL, 2017.
    .. [2] add reference
    """

    def build_sample(self, model, n_samples, method):
        """
        Creates the Monte-Carlo samples for the variables.

        This method creates the input samples and evaluate them through the 
        model to create the output samples.

        Parameters
        ----------
        model : ProbabilisticModel's instance.
            An object containing all the necessary information about the model.

        n_samples : int
            The sampling size for the Monte-Carlo estimation.

        method : str
            Parameter allowing to build the suitable samples for the estimation.
            if method == 'Min_in_Leaves', just need X1, Y1
            if method == 'Weighted_Min', 'Weighted_Min_with_complete_forest' or 
                         'Kernel_Min', need X1,Y1 and X2
        """

        assert isinstance(model, ProbabilisticModel), \
            "The model should be an instance of the class ProbabilisticModel. Given %s" %(type(model),)
        assert isinstance(n_samples, (int, np.integer)), \
            "The number of sample should be an integer."
        assert n_samples > 0, \
            "The number of sample should be positive: %d<0" % (n_samples,)
        assert isinstance(method, str), \
            "The parameter 'method' should be a string. Given %s" % (type(method),)
        assert method in ("Kernel_Min", "Min_in_Leaves", "Weighted_Min", 
                          "Weighted_Min_with_complete_forest"), "Given method is not known: %s" % (method,)

        input_distribution = model.input_distribution

        # Simulate the samples
        self.input_samples_1 = np.asarray(input_distribution.getSample(n_samples),
                                          order='F')
        self.output_samples_1 = model(self.input_samples_1)

        if method == "Min_in_Leaves":
            self.input_samples_2 = None
        else:
            self.input_samples_2 = np.asarray(input_distribution.getSample(n_samples),
                                      order='F')
        
        self._n_samples = n_samples
        self._dim = model.dim
        self._model = model
        self._method_name = method

    def feed_sample(self, X1, Y1, method, X2=None):
        """
        The user provides the Monte-Carlo samples to compute the Qosa indices.

        Parameters
        ----------
        X1 : array-like of shape = [n_samples, n_features]
            Input samples used to compute the QOSA indices
        
        Y1 : array-like, shape = [n_samples]
            Output samples used to compute the QOSA indices

        X2 : array-like of shape = [n_samples, n_features], default=None
            Input samples used to compute the QOSA indices

        method : str
            Parameter allowing to build the suitable sample for the estimation.
            if method == 'Min_in_Leaves', just need X1, Y1
            if method == 'Weighted_Min', 'Weighted_Min_with_complete_forest', need X1,Y1 and X2
        """

        assert isinstance(method, str), \
            "The parameter 'method' should be a string. Given %s" %(type(method),)
        if method in ("Weighted_Min", "Weighted_Min_with_complete_forest") and X2 is None:
            raise ValueError("You need to feed the parameter 'X2' if you choose "
                             "'Weighted_Min' or 'Weighted_Min_with_complete_forest' for method.")
        if method == "Kernel_Min":
            raise ValueError("The method 'Kernel_Min' needs the density of the marginals "
                             "during the estimation. Thus, you can use this method only "
                             "with the function 'build_sample' which allows you to pass "
                             "the input distribution.")
        assert method in ("Min_in_Leaves", "Weighted_Min", "Weighted_Min_with_complete_forest"), \
            "Only 'Min_in_Leaves', 'Weighted_Min' or 'Weighted_Min_with_complete_forest' are allowed like method."

        # Transformation of the data in np.array
        X1 = np.asarray(X1, order='F')
        Y1 = np.asarray(Y1)
        if X2 is not None:
            X2 = np.asarray(X2, order='F')

        # Validate or convert the input data
        try:
            X1, Y1 = check_X_y(X1, Y1, dtype=DTYPE, y_numeric=True)
        except ValueError as instance_ValueError:
            if X1.shape[0] == X1.size:
                X1 = X1.reshape(-1, 1, order='F')
                X1, Y1 = check_X_y(X1, Y1, dtype=DTYPE, y_numeric=True)
            else:
                raise ValueError(instance_ValueError)

        if X2 is not None:
            X2 = check_array(X2, dtype=DTYPE)
            assert X1.shape[1] == X2.shape[1], "X1 and X2 do not have the same number of features."                         

        self.input_samples_1 = X1
        self.input_samples_2 = X2
        self.output_samples_1 = Y1

        # Informations about the problem dimension
        self._n_samples, self._dim = X1.shape
        self._model = None
        self._method_name = method 

    def compute_indices(self, method):
        """
        Method to compute the QOSA indices from the previously created samples
        thanks to 'build_sample' or 'feed_sample'.
        
        Parameters
        ----------
        method : object
            Choose which method to use for computing the expected value of the check
            function among the classes following:
            - the forest methods: 'qosa_Min__Min_in_Leaves', 'qosa_Min__Weighted_Min', 
                                  'qosa_Min__Weighted_Min_with_complete_forest'   
            - the kernel method : 'qosa_Min__Kernel_Min'

        Returns
        -------
        results : SensitivityResults instance
            The computed Qosa indices.
            If the alpha levels are provided in no particular order,
            e.g. alpha=[0.5, 0.1, 0.9], the returned values for each variable 
            are sorted according to alpha levels, i.e. [0.1, 0.5, 0.9].
        """

        assert isinstance(method, (qosa_Min__Kernel_Min, qosa_Min__Min_in_Leaves, 
                                   qosa_Min__Weighted_Min, qosa_Min__Weighted_Min_with_complete_forest)), \
                          "The provided class is not known. Given %s" % (type(method), )

        if self._method_name == "Min_in_Leaves" and method.name in ("Kernel_Min",
            "Weighted_Min", "Weighted_Min_with_complete_forest"):
            raise ValueError("The method that you have chosen it's not adapted with "
                             " the sample you build/provided previously.")

        alpha = np.asarray(method.alpha).astype(dtype=np.float64)
        alpha.sort()
        dim = self._dim

        X1 = self.input_samples_1
        Y1 = self.output_samples_1
        X2 = self.input_samples_2

        if method.name == "Kernel_Min":
            results_indice_func = _qosa_compute_mean_Kernel_Min(
                                        X1=X1,
                                        Y1=Y1,
                                        X2=X2,
                                        input_distribution=self._model.input_distribution,
                                        dim=dim,
                                        alpha=alpha,
                                        bandwidth=method.bandwidth,
                                        optim_by_CV=method.optim_by_CV,
                                        n_fold=method.n_fold)
        elif method.name == "Min_in_Leaves":
            results_indice_func = _qosa_compute_mean_Min_in_Leaves(
                                        X1=X1,
                                        Y1=Y1,
                                        dim=dim,
                                        alpha=alpha,
                                        n_estimators=method.n_estimators,
                                        min_samples_leaf=method.min_samples_leaf,
                                        used_bootstrap_samples=method.used_bootstrap_samples,
                                        optim_by_CV=method.optim_by_CV,
                                        n_fold=method.n_fold,
                                        random_state_Forest=method.random_state_Forest)
        
        elif method.name == "Weighted_Min":
            results_indice_func = _qosa_compute_mean_Weighted_Min(
                                        X1=X1,
                                        Y1=Y1,
                                        X2=X2,
                                        dim=dim,
                                        alpha=alpha,
                                        n_estimators=method.n_estimators,
                                        min_samples_leaf=method.min_samples_leaf,
                                        used_bootstrap_samples=method.used_bootstrap_samples,
                                        optim_by_CV=method.optim_by_CV,
                                        n_fold=method.n_fold,
                                        random_state_Forest=method.random_state_Forest)
        elif method.name == "Weighted_Min_with_complete_forest":
            results_indice_func = _qosa_compute_mean_Weighted_Min_with_complete_forest(
                                        X1=X1,
                                        Y1=Y1,
                                        X2=X2,
                                        dim=dim,
                                        alpha=alpha,
                                        n_estimators=method.n_estimators,
                                        min_samples_leaf=method.min_samples_leaf,
                                        used_bootstrap_samples=method.used_bootstrap_samples,
                                        optim_by_CV=method.optim_by_CV,
                                        n_fold=method.n_fold,
                                        random_state_Forest=method.random_state_Forest)

        qosa_indices = results_indice_func[0]
        optimal_parameter_by_CV = results_indice_func[1]

        if isinstance(self._model, ProbabilisticModel):
            self._model.alpha = alpha
            true_qosa_indices = self._model.qosa_indices
        else:
            true_qosa_indices = None

        results = SensitivityResults(
                                alpha=alpha,
                                qosa_indices_estimates=qosa_indices,
                                true_qosa_indices=true_qosa_indices,
                                dim=dim,
                                optim_by_CV=method.optim_by_CV,
                                optimal_parameter_by_CV=optimal_parameter_by_CV,
                                method=method.name)
        return results


_QUANTILE_ESTIMATORS_FOREST  = ('Averaged_Quantile', 'Weighted_CDF')
_QUANTILE_ESTIMATORS_KERNEL = ('Kernel_CDF')




# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------
#
# Private ancillary functions for the previous classes
#
# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------

##########################################
#                                        #
# For the class QuantileBasedQosaIndices #
#                                        #
##########################################

def _qosa_forest_compute_quantile(X1,
                                  X2, 
                                  Y1,
                                  Y2,
                                  dim,
                                  alpha,
                                  method,
                                  n_estimators,
                                  min_samples_leaf,
                                  used_bootstrap_samples,
                                  optim_by_CV,
                                  CV_strategy,
                                  n_fold,
                                  random_state_Forest):
    """
    Compute the qosa indices with the Random Forest method by plugging the
    conditional quantiles.
    """
    
    n_alphas = alpha.shape[0]
    
    # Compute the denominator of the indices
    alpha_quantile = np.percentile(Y1, q=alpha*100)
    denominator = _averaged_check_function_alpha_array(
                                                Y1.reshape(-1,1) - alpha_quantile,
                                                alpha)

    # Compute the numerator of the indices for each variable
    numerator = np.empty((dim, n_alphas), dtype=np.float64)
    min_samples_leaf_by_dim_and_alpha = np.empty((n_alphas, dim), dtype=np.uint32) if optim_by_CV else None
    for i in range(dim):
        X1_i = X1[:, i]
        X2_i = X2[:, i]

        if optim_by_CV:
            _, optim_min_samples_leaf = cross_validation_forest(
                                            X=X2_i,
                                            y=Y2,
                                            alpha=alpha,
                                            min_samples_leaf=min_samples_leaf,
                                            method=method,
                                            n_estimators=n_estimators,
                                            used_bootstrap_samples=used_bootstrap_samples,
                                            CV_strategy=CV_strategy,
                                            n_fold=n_fold)

            for j, alpha_j in enumerate(alpha):
                min_samples_leaf_by_dim_and_alpha[j, i] = optim_min_samples_leaf[j]
                quantForest = QuantileRegressionForest(
                                    n_estimators=n_estimators,
                                    min_samples_split=min_samples_leaf_by_dim_and_alpha[j, i]*2,
                                    min_samples_leaf=min_samples_leaf_by_dim_and_alpha[j, i],
                                    random_state=random_state_Forest)
                quantForest.fit(X2_i, Y2)
                conditional_quantiles = quantForest.predict_quantile(
                                                X=X1_i,
                                                alpha=alpha_j,
                                                method=method,
                                                used_bootstrap_samples=used_bootstrap_samples)
                numerator[i, j] = _averaged_check_function_alpha_float_parallel(
                                                        Y1 - conditional_quantiles.ravel(),
                                                        alpha_j)
        else:
            quantForest = QuantileRegressionForest(n_estimators=n_estimators,
                                                   min_samples_split=min_samples_leaf*2,
                                                   min_samples_leaf=min_samples_leaf,
                                                   random_state=random_state_Forest)
            quantForest.fit(X2_i, Y2)
            conditional_quantiles = quantForest.predict_quantile(
                                                X=X1_i,
                                                alpha=alpha,
                                                method=method,
                                                used_bootstrap_samples=used_bootstrap_samples)
            numerator[i, :] = _averaged_check_function_alpha_array(
                                            Y1.reshape(-1,1) - conditional_quantiles,
                                            alpha)

    qosa_indices = 1 - numerator/denominator

    return qosa_indices.T, min_samples_leaf_by_dim_and_alpha


def _qosa_kernel_compute_quantile(X1,
                                  X2,
                                  Y1,
                                  Y2,
                                  dim,
                                  alpha,
                                  bandwidth,
                                  optim_by_CV,
                                  n_fold):
    """
    Compute the qosa indices with the estimation method proposed by Ibrahima Niang
    by plugging the conditional quantiles
    """
    
    n_alphas = alpha.shape[0]
    p = 1./(1 - alpha)

    # Computation of the expectation of the output
    expectation_Y = (Y1.mean() + Y2.mean())*0.5
    
    # Compute the denominator of the indices
    alpha_quantile = np.percentile(Y2, q=alpha*100)
    denominator = p*(Y1.reshape(-1, 1)*(Y1.reshape(-1, 1) > alpha_quantile)).mean(axis=0) - expectation_Y

    # Compute the numerator of the indices for each variable
    numerator = np.empty((dim, n_alphas), dtype=np.float64)
    bandwidth_by_dim_and_alpha = np.empty((n_alphas, dim), dtype=np.float64) if optim_by_CV else None
    for i in range(dim):
        X1_i = X1[:, i]
        X2_i = X2[:, i]

        if optim_by_CV:
            _, optim_bandwidth = cross_validation_kernel(X=X2_i,
                                                         y=Y2,
                                                         alpha=alpha,
                                                         bandwidth=bandwidth,
                                                         n_fold=n_fold)

            uquantKernel = UnivariateQuantileRegressionKernel()
            uquantKernel.fit(X2_i, Y2)
            for j, alpha_j in enumerate(alpha):
                bandwidth_by_dim_and_alpha[j, i] = optim_bandwidth[j]
                conditional_quantiles = uquantKernel.predict(
                                                    X=X1_i,
                                                    alpha=alpha_j,
                                                    bandwidth=bandwidth_by_dim_and_alpha[j, i])
                numerator[i, j] = p[j]*(Y1.reshape(-1, 1)*(Y1.reshape(-1, 1) > conditional_quantiles)).mean(axis=0) - expectation_Y
        else:
            uquantKernel = UnivariateQuantileRegressionKernel()
            uquantKernel.fit(X2_i, Y2)
            conditional_quantiles = uquantKernel.predict(X=X1_i,
                                                         alpha=alpha,
                                                         bandwidth=bandwidth)
            numerator[i, :] = p*(Y1.reshape(-1, 1)*(Y1.reshape(-1, 1) > conditional_quantiles)).mean(axis=0) - expectation_Y

    qosa_indices = 1 - numerator/denominator

    return qosa_indices.T, bandwidth_by_dim_and_alpha


#########################################
#                                       #
# For the class MinimumBasedQosaIndices #
#                                       #
#########################################

def _qosa_compute_mean_Kernel_Min(X1,
                                  Y1,
                                  X2,
                                  input_distribution,
                                  dim,
                                  alpha,
                                  bandwidth,
                                  optim_by_CV,
                                  n_fold):
    """
    Compute the qosa indices with the estimation method proposed by Thomas Browne.
    """
    
    n_alphas = alpha.shape[0]

    # Compute the denominator of the indices
    denominator = _compute_first_term_Browne_estimate(Y1, alpha)

    # Compute the numerator of the indices for each variable
    numerator = np.empty((dim, n_alphas), dtype=np.float64)
    bandwidth_by_dim_and_alpha = np.empty((n_alphas, dim), dtype=np.float64) if optim_by_CV else None
    for i in range(dim):
        X1_i = X1[:, i]
        X2_i = X2[:, i]
        density_i = np.array(input_distribution.getMarginal(i).computePDF(X2_i.reshape(-1, 1))).ravel()

        if optim_by_CV:
            _, optim_bandwidth = cross_validation_kernel(X=X1_i,
                                                         y=Y1,
                                                         alpha=alpha,
                                                         bandwidth=bandwidth,
                                                         n_fold=n_fold)

            min_expectation = MinimumConditionalExpectedCheckFunctionKernel()
            min_expectation.fit(X1_i, Y1)
            for j, alpha_j in enumerate(alpha):
                bandwidth_by_dim_and_alpha[j, i] = optim_bandwidth[j]
                numerator[i, j] = min_expectation.predict(X=X2_i,
                                                          X_density=density_i,
                                                          alpha=alpha_j,
                                                          bandwidth=bandwidth_by_dim_and_alpha[j, i]
                                                          ).mean()
        else:
            min_expectation = MinimumConditionalExpectedCheckFunctionKernel()
            min_expectation.fit(X1_i, Y1)
            numerator[i, :] = min_expectation.predict(X=X2_i,
                                                      X_density=density_i,
                                                      alpha=alpha,
                                                      bandwidth=bandwidth).mean(axis=0)

    qosa_indices = 1 - numerator/denominator

    return qosa_indices.T, bandwidth_by_dim_and_alpha

# -------------------------------------------------------------------------
# Numba function computing the denominator of the QOSA index for the method 
# proposed by Thomas
# -------------------------------------------------------------------------

@nb.njit("float64[:](float64[:], float64[:])", nogil=True, cache=True, parallel=False)
def _compute_first_term_Browne_estimate(output_samples, alpha):
    
    n_alphas = alpha.shape[0]
    argsort_output_samples = np.argsort(output_samples)
    expectation = np.empty(n_alphas, dtype=np.float64)
    
    j=0
    for i, alpha_temp in enumerate(alpha):
        if j < output_samples.shape[0]:
            ans = _averaged_check_function_alpha_float_unparallel(
                                output_samples - output_samples[argsort_output_samples[j]],
                                alpha_temp) 
        else:
            for k in range(i, n_alphas):
                expectation[k] = _averaged_check_function_alpha_float_unparallel(
                                output_samples - output_samples[argsort_output_samples[-1]],
                                alpha[k])
            break
                
        if i == 0:
            j=1
        else:
            j+=1
        while(j < output_samples.shape[0]):
            temp = _averaged_check_function_alpha_float_unparallel(
                                output_samples - output_samples[argsort_output_samples[j]],
                                alpha_temp)
            if(temp <= ans):
                ans = temp
                j+=1
            else:
                j-=1
                break
        expectation[i] = ans
    
    return expectation


def _qosa_compute_mean_Min_in_Leaves(X1,
                                     Y1,
                                     dim,
                                     alpha,
                                     n_estimators,
                                     min_samples_leaf,
                                     used_bootstrap_samples,
                                     optim_by_CV,
                                     n_fold,
                                     random_state_Forest):
    """
    Compute the qosa indices with the Random Forest method by getting the 
    minimum in each leaf of each tree
    """

    n_alphas = alpha.shape[0]
    
    # Compute the denominator of the indices
    alpha_quantile = np.percentile(Y1, q=alpha*100)
    denominator = _averaged_check_function_alpha_array(
                                                Y1.reshape(-1,1) - alpha_quantile,
                                                alpha)

    # Compute the numerator of the indices for each variable
    numerator = np.empty((dim, n_alphas, 2), dtype=np.float64) # 2 for the classical and weighted mean
    min_samples_leaf_by_dim_and_alpha = np.empty((n_alphas, dim), dtype=np.uint32) if optim_by_CV else None
    for i in range(dim):
        X1_i = X1[:, i]

        if optim_by_CV:
            _, optim_min_samples_leaf = cross_validation_forest(
                                                X=X1_i,
                                                y=Y1,
                                                alpha=alpha,
                                                min_samples_leaf=min_samples_leaf,
                                                method="Weighted_CDF",
                                                n_estimators=n_estimators,
                                                used_bootstrap_samples=False,
                                                CV_strategy="K_Fold",
                                                n_fold=n_fold)

            for j,alpha_j in enumerate(alpha):
                min_samples_leaf_by_dim_and_alpha[j, i] = optim_min_samples_leaf[j]
                averagedForest = MinimumConditionalExpectedCheckFunctionWithLeaves(
                                    n_estimators=n_estimators,
                                    min_samples_split=min_samples_leaf_by_dim_and_alpha[j, i]*2,
                                    min_samples_leaf=min_samples_leaf_by_dim_and_alpha[j, i],
                                    random_state=random_state_Forest)
                averagedForest.fit(X1_i, Y1)
                numerator[i, j, :] = averagedForest.predict(
                                                alpha_j,
                                                used_bootstrap_samples=used_bootstrap_samples)
        else:
            averagedForest = MinimumConditionalExpectedCheckFunctionWithLeaves(
                                                n_estimators=n_estimators,
                                                min_samples_split=min_samples_leaf*2,
                                                min_samples_leaf=min_samples_leaf,
                                                random_state=random_state_Forest)
            averagedForest.fit(X1_i, Y1)
            numerator[i, :, :] = averagedForest.predict(
                                                alpha,
                                                used_bootstrap_samples=used_bootstrap_samples)

    qosa_indices = 1 - numerator/denominator[:,np.newaxis]

    return qosa_indices.T, min_samples_leaf_by_dim_and_alpha


def _qosa_compute_mean_Weighted_Min(X1,
                                    Y1,
                                    X2,
                                    dim,
                                    alpha,
                                    n_estimators,
                                    min_samples_leaf,
                                    used_bootstrap_samples,
                                    optim_by_CV,
                                    n_fold,
                                    random_state_Forest):
    """
    Compute the qosa indices with the Random Forest method by getting the 
    minimum with a weithed mean.
    """

    n_alphas = alpha.shape[0]
    
    # Compute the denominator of the indices
    alpha_quantile = np.percentile(Y1, q=alpha*100)
    denominator = _averaged_check_function_alpha_array(
                                                Y1.reshape(-1,1) - alpha_quantile,
                                                alpha)

    # Compute the numerator of the indices for each variable
    numerator = np.empty((dim, n_alphas), dtype=np.float64)
    min_samples_leaf_by_dim_and_alpha = np.empty((n_alphas, dim), dtype=np.uint32) if optim_by_CV else None
    for i in range(dim):
        X1_i = X1[:, i]
        X2_i = X2[:, i]

        if optim_by_CV:
            _, optim_min_samples_leaf = cross_validation_forest(
                                            X=X1_i,
                                            y=Y1,
                                            alpha=alpha,
                                            min_samples_leaf=min_samples_leaf,
                                            method="Weighted_CDF",
                                            n_estimators=n_estimators,
                                            used_bootstrap_samples=False,
                                            CV_strategy="K_Fold",
                                            n_fold=n_fold)

            for j, alpha_j in enumerate(alpha):
                min_samples_leaf_by_dim_and_alpha[j, i] = optim_min_samples_leaf[j]
                min_expectation = MinimumConditionalExpectedCheckFunctionWithWeights(
                                    n_estimators=n_estimators,
                                    min_samples_split=min_samples_leaf_by_dim_and_alpha[j, i]*2,
                                    min_samples_leaf=min_samples_leaf_by_dim_and_alpha[j, i],
                                    random_state=random_state_Forest)
                min_expectation.fit(X1_i, Y1)
                numerator[i, j] = min_expectation.predict(X=X2_i,
                                                          alpha=alpha_j,
                                                          used_bootstrap_samples=used_bootstrap_samples
                                                          ).mean()
        else:
            min_expectation = MinimumConditionalExpectedCheckFunctionWithWeights(
                                            n_estimators=n_estimators,
                                            min_samples_split=min_samples_leaf*2,
                                            min_samples_leaf=min_samples_leaf,
                                            random_state=random_state_Forest)
            min_expectation.fit(X1_i, Y1)
            numerator[i, :] = min_expectation.predict(X=X2_i, 
                                                      alpha=alpha,
                                                      used_bootstrap_samples=used_bootstrap_samples
                                                      ).mean(axis=0)

    qosa_indices = 1 - numerator/denominator

    return qosa_indices.T, min_samples_leaf_by_dim_and_alpha


def _qosa_compute_mean_Weighted_Min_with_complete_forest(X1,
                                                         Y1,
                                                         X2,
                                                         dim,
                                                         alpha,
                                                         n_estimators,
                                                         min_samples_leaf,
                                                         used_bootstrap_samples,
                                                         optim_by_CV,
                                                         n_fold,
                                                         random_state_Forest):
    """
    Compute the qosa indices with the Random Forest method by getting the 
    minimum with a weithed  mean.
    """
    
    n_alphas = alpha.shape[0]
    n_samples = X2.shape[0]
   
    # Construction of the forest
    forest = BaseEstimatorForest(n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_leaf*2,
                                 random_state=random_state_Forest,
                                 n_jobs=-1)
    forest.fit(X1, Y1)
    forest.n_jobs = 1 # Back to serial execution of forest methods

    if used_bootstrap_samples:
            _, inbag_samples = forest._compute_inbag_samples()
    else:
        inbag_samples = np.empty((1, 1), dtype=np.uint32)

    # Compute the numerator of the indices for each variable
    numerator = np.zeros((dim, n_alphas), dtype=np.float64)
    
    with Pool(initializer=initializer, initargs=(X2, forest, inbag_samples, alpha, used_bootstrap_samples)) as pool:

        results = pool.imap_unordered(task, itertools.product(range(dim), range(n_samples)))

        for i, minimum_conditional_expectation in results:
            numerator[i, :] += minimum_conditional_expectation

        numerator /= n_samples

    # Compute the denominator of the indices
    alpha_quantile = np.percentile(Y1, q=alpha*100)
    denominator = _averaged_check_function_alpha_array(
                                                Y1.reshape(-1,1) - alpha_quantile,
                                                alpha)

    qosa_indices = 1 - numerator/denominator
    min_samples_leaf_by_dim_and_alpha = None

    return qosa_indices.T, min_samples_leaf_by_dim_and_alpha

# ---------------------------------
# Functions to multiprocessing.Pool
# ---------------------------------

# Function to provide the data to each process at the beginning
def initializer(_X2, _forest, _inbag_samples, _alpha, _used_bootstrap_samples):
    task.X2 = _X2
    task.forest = _forest
    task.inbag_samples = _inbag_samples
    task.alpha = _alpha
    task.used_bootstrap_samples = _used_bootstrap_samples

# Task carried out by each process
def task(args):
    i, j = args
    return i, _compute_conditional_minimum_expectation_by_observation_for_numerator_qosa_index_by_variable(
                                                                        task.X2,
                                                                        i,
                                                                        j,
                                                                        task.forest,
                                                                        task.inbag_samples,
                                                                        task.alpha,
                                                                        task.used_bootstrap_samples)

def _compute_conditional_minimum_expectation_by_observation_for_numerator_qosa_index_by_variable(
                                                                             X2, 
                                                                             feature,
                                                                             idx_min_obs,
                                                                             forest,
                                                                             inbag_samples,
                                                                             alpha,
                                                                             used_bootstrap_samples):
    X2_temp = X2.copy()
    X2_temp[:, feature] = X2[idx_min_obs, feature]
    
    X2_temp_nodes = forest.apply(X2_temp)
    _, idx_unique_X2_temp_nodes, counts_unique_X2_temp_nodes = np.unique(
                                                                     X2_temp_nodes, 
                                                                     axis=0,
                                                                     return_index=True,
                                                                     return_inverse=False,
                                                                     return_counts=True)
        
    minimum_expectation = _compute_conditional_minimum_expectation_with_averaged_weights_of_complete_forest(
                                                    forest._output_samples,
                                                    forest._samples_nodes,
                                                    inbag_samples,
                                                    X2_temp_nodes[idx_unique_X2_temp_nodes],
                                                    counts_unique_X2_temp_nodes,
                                                    alpha,
                                                    used_bootstrap_samples)
    return minimum_expectation

@nb.njit("float64[:](float64[:], int64[:,:], uint32[:,:], int64[:,:], int64[:], float64[:], boolean)", nogil=True, cache=True, parallel=False)
def _compute_conditional_minimum_expectation_with_averaged_weights_of_complete_forest(
                                                                             output_samples,
                                                                             samples_nodes,
                                                                             inbag_samples,
                                                                             unique_X2_temp_nodes,
                                                                             counts_unique_X2_temp_nodes,
                                                                             alpha,
                                                                             used_bootstrap_samples):
    n_alphas = alpha.shape[0]    
    n_samples = output_samples.shape[0]
    n_unique_X2_temp_nodes = unique_X2_temp_nodes.shape[0]
    argsort_output_samples = np.argsort(output_samples)
    mean_weights = np.zeros(n_samples, dtype=np.float64)
    minimum_expectation = np.empty(n_alphas, dtype=np.float64)
    
    # For each observation, complete the averaged weight thanks to the forest built with all variables
    for i in range(n_unique_X2_temp_nodes):
        if used_bootstrap_samples:
            # Compute the Boostrap Data based conditional weights associated to each individual
            weight = _compute_weight_with_bootstrap_data(samples_nodes,
                                                         inbag_samples,
                                                         unique_X2_temp_nodes[i, :])
        else:
            # Compute the Original Data based conditional weights associated to each individual
            weight = _compute_weight_with_original_data(samples_nodes, unique_X2_temp_nodes[i, :])
        
        mean_weights += weight*counts_unique_X2_temp_nodes[i]
    mean_weights /= n_samples
    
    k=0
    for j, alpha_temp in enumerate(alpha):
        if k < n_samples:
            ans = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                 output_samples,
                                                 mean_weights,
                                                 output_samples[argsort_output_samples[k]],
                                                 alpha_temp)
        else:
            for l in range(j, n_alphas):
                minimum_expectation[l] = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                     output_samples,
                                                     mean_weights,
                                                     output_samples[argsort_output_samples[-1]],
                                                     alpha[l])
            break
                
        if j == 0: 
            k=1
        else:
            k+=1
        while(k < n_samples):
            temp = _averaged_check_function_alpha_float_unparallel_product_weight(
                                             output_samples,
                                             mean_weights,
                                             output_samples[argsort_output_samples[k]],
                                             alpha_temp)
            if(temp <= ans):
                ans = temp
                k+=1
            else:
                k-=1
                break
        minimum_expectation[j] = ans
    
    return minimum_expectation