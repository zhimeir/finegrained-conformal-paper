# -*- coding: utf-8 -*-

"""
Add docstring of the module
"""


import openturns as ot
import numpy as np
import pandas as pd


__all__ = ['BaseIndices', 'SensitivityResults']


class BaseIndices(object):
    """
    Base class for sensitivity indices.

    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        An OpenTURNS distribution object.
    """

    def __init__(self, input_distribution):
        self.input_distribution = input_distribution
        self.indice_func = None

    @property
    def input_distribution(self):
        """
        The OpenTURNS input distribution.
        """
        return self._input_distribution

    @input_distribution.setter
    def input_distribution(self, dist):
        assert isinstance(dist, ot.DistributionImplementation), \
            "The distribution should be an OpenTURNS Distribution object. Given %s" % (type(dist))
        self._input_distribution = dist

    @property
    def indice_func(self):
        """
        Function to estimate the indice.
        """
        return self._indice_func

    @indice_func.setter
    def indice_func(self, func):
        assert callable(func) or func is None, \
            "Indice function should be callable or None."
        self._indice_func = func

    @property
    def dim(self):
        """
        The input dimension.
        """
        return self._input_distribution.getDimension()


class SensitivityResults(object):
    """
    Class to gather the Sensitivity Analysis results
    """

    def __init__(self,
                 alpha=None,
                 qosa_indices_estimates=None,
                 true_qosa_indices=None,
                 dim=None,
                 optim_by_CV=None,
                 optimal_parameter_by_CV=None,
                 method=None):

        self.alpha = alpha
        self.qosa_indices_estimates = qosa_indices_estimates
        self.true_qosa_indices = true_qosa_indices
        self.dim = dim
        self.optim_by_CV = optim_by_CV
        self.optimal_parameter_by_CV = optimal_parameter_by_CV
        self.method = method
        self.var_names = None

    @property
    def var_names(self):
        """
        Names of the variables inside the model
        """

        if self._var_names is None and self.dim is not None:
            dim = self.dim
            columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
            return columns
        else:
            return self._var_names

    @var_names.setter
    def var_names(self, names):
        self._var_names = names

    @property
    def true_indices(self):
        """
        The true sensitivity results.
        """

        if self.alpha is None:
            raise AttributeError("You should specify the alpha attribute in"
                                 " order to know at which alpha order matches"
                                 " the qosa indices.")
        else:
            alpha = self.alpha

        if self.true_qosa_indices is not None:
            index_columns = ['$X_{%d}$' % (i+1) for i in range(self.dim)]
            index_rows = self.alpha

            df_indices = pd.DataFrame(index=index_rows, columns=index_columns)
            df_indices.iloc[:] = self.true_qosa_indices
            indices = df_indices
        else:
            indices = None 

        return indices

    @property
    def qosa_indices_estimates(self):
        """
        The Qosa indices estimation.
        """
        return self._qosa_indices_estimates

    @qosa_indices_estimates.setter
    def qosa_indices_estimates(self, indices):
        if self.alpha is None:
            raise AttributeError("You should first specify the alpha attribute in order" 
                                 " to know at which alpha order matches the qosa indices.")
        self._qosa_indices_estimates = np.asarray(indices)


class SensitivityResults_QOSE(object):
    """
    Class to gather the sensitivity analysis results regarding QOSE indices
    """
    def __init__(self,
                 alpha=None,
                 qose_indices=None,
                 true_qose_indices=None,
                 qose_indices_SE=None,
                 estimation_method=None):
        self.dim = None
        self.n_alphas = None
        self.n_boot = None
        self.var_names = None
        self.alpha = alpha
        self.qose_indices = qose_indices
        self.true_qose_indices = true_qose_indices
        self.qose_indices_SE = qose_indices_SE
        self.estimation_method = estimation_method

    @property
    def var_names(self):
        """
        Names of the variables inside the model
        """
        if self._var_names is None and self.dim is not None:
            dim = self.dim
            columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
            return columns
        else:
            return self._var_names

    @var_names.setter
    def var_names(self, names):
        self._var_names = names

    @property
    def qose_indices(self):
        """
        The Shapley indices estimations.
        """
        if self._qose_indices is not None:
            if self.n_boot == 1:
                return self._qose_indices.squeeze()
            else:
                return self._qose_indices

    @qose_indices.setter
    def qose_indices(self, indices):
        if indices is not None:
            indices = np.asarray(indices)
            self.dim, self.n_alphas, self.n_boot = indices.shape
        self._qose_indices = indices

    @property
    def qose_indices_SE(self):
        """
        The shapley sensitivity estimation for c.i.
        """
        if self._qose_indices_SE is not None:
            return self._qose_indices_SE

    @qose_indices_SE.setter
    def qose_indices_SE(self, indices):
        if indices is not None:
            indices = np.asarray(indices)
        self._qose_indices_SE = indices

    @property
    def full_qose_indices(self):
        """
        """
        if np.isnan(self._qose_indices).all():
            raise ValueError('The value is not registered')
        return self._qose_indices

    