# -*- coding: utf-8 -*-

"""
Add docstring of the module
"""


import numpy as np
import openturns as ot
import pandas as pd


__all__ = ['Model', 'ProbabilisticModel']


class Model(object):
    """
    A model that evaluates a given function.
    This class aims to gather the informations of a model function.

    Parameters
    ----------
    model_func : callable
        The model function.

    name : str
        The function name.
    """

    def __init__(self, model_func, name='Custom'):
        self.model_func = model_func
        self.name = name

    def __call__(self, x):
        y = self._model_func(x)
        return y

    @property
    def model_func(self):
        """
        The model function.
        """
        return self._model_func

    @model_func.setter
    def model_func(self, func):
        if func is not None:
            assert callable(func), "The function should be callable"
        self._model_func = func


class ProbabilisticModel(Model):
    """
    A probabilistic model that evaluates a given function.
    This class aims to gather the informations of a function and its 
    probabilistic input distribution.

    Parameters
    ----------
    model_func : callable
        The model function.

    input_distribution : ot.DistributionImplementation
        The probabilistic input distribution.

    name : str, default='Custom'
        The function name.

    qosa_indices : list, array or None, default=None
        The true QOSA indices if you know the true ones.

    alpha : array-like of shape = [n_alphas]
        The order of the QOSA indices where you know the true values.
    """

    def __init__(self,
                 model_func,
                 input_distribution=None,
                 name='Custom',
                 qosa_indices=None,
                 alpha=None):
        
        if (input_distribution is None) and (qosa_indices is not None):
            raise AttributeError("If you know the true qosa indices, you should"
                                 " specify the associated input distribution")
        if (alpha is None) and (qosa_indices is not None):
            raise AttributeError("If you know the true qosa indices, you should"
                                 " specify the associated alpha order where you"
                                 " know the true values.")

        super(ProbabilisticModel, self).__init__(model_func=model_func, name=name)
        self.input_distribution = input_distribution        
        self.alpha = alpha
        self.qosa_indices = qosa_indices

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
        self._dim = dist.getDimension()
        self._margins = [dist.getMarginal(i) for i in range(self._dim)]
        self._copula = dist.getCopula()
        self._input_distribution = dist

    @property
    def dim(self):
        """
        The problem dimension.
        """
        return self._dim

    @property
    def margins(self):
        """
        The input marginal distributions.
        """
        return self._margins

    @margins.setter
    def margins(self, margins):
        _check_margins(margins, self._dim)
        self._input_distribution = ot.ComposedDistribution(margins, self._copula)
        self._margins = margins

    @property
    def copula(self):
        """
        The input copula.
        """
        return self._copula

    @copula.setter
    def copula(self, copula):
        _check_copula(copula, self._dim)
        self._input_distribution = ot.ComposedDistribution(self._margins, copula)
        self._copula = copula

    @property
    def copula_parameters(self):
        """
        The copula parameters.
        """
        return self._copula_parameters

    @copula_parameters.setter
    def copula_parameters(self, params):
        copula = self._copula
        copula.setParameter(params)
        self.copula = copula
        self._copula_parameters = params

    @property
    def alpha(self):
        """
        The alpha order where to assess the qosa indices.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha_values):
        if alpha_values is not None:
            if isinstance(alpha_values, (int, np.integer, float, np.floating)):
                alpha_values = [alpha_values]
            alpha_values = np.asarray(alpha_values)
        self._alpha = alpha_values

    @property
    def qosa_indices(self):
        """
        The true qosa indices.
        """
        return self._qosa_indices

    @qosa_indices.setter
    def qosa_indices(self, indices):
        if indices is not None:
            if self._alpha is None:
                raise AttributeError("You should specify the alpha attribute in"
                                     " order to know at which alpha order"
                                     " matches the qosa indices that you provide.")
            else:
                indices = _check_indices(indices, self._dim, self._alpha.shape[0])
        self._qosa_indices = indices

    @property
    def df_indices(self):
        """
        A dataframe of the true indices.

        Returns
        -------
        indices : dataframe
            The dataframe of the registered sensitivity indices.
        """

        if self._qosa_indices is not None:
            index_columns = ['$X_{%d}$' % (i+1) for i in range(self._dim)]
            index_rows = self._alpha

            df_indices = pd.DataFrame(index=index_rows, columns=index_columns)
            df_indices.iloc[:] = self._qosa_indices
        else:
            indices = None
            print("There is no true indices.")
        return indices


def _check_indices(indices, dim, n_alphas):
    n_al = indices.shape[0] # Number of alpha
    try:  # Works if indices is a 2d array
        d = indices.shape[1]  # Dimension of the array
        if d != dim or n_al != n_alphas: # If the dimension is not correct
            if n_al == dim and d == n_alphas: 
                indices = indices.T
            else:  # Error
                raise ValueError("indices dimension is different from the"
                                 "model dimension : %d (indices) != %d (model)" 
                                 % (d, dim))
    except:  # Its a vector
        d = 1
        if d != dim:  # If the dimension is not correct
            if n_al == dim and n_alphas==1:
                d = n_al
                n_al = 1
                indices = indices.reshape(n_al, d)
            else:  # Error
                raise ValueError("indices dimension is different from the"
                                 "model dimension : %d (indices) != %d (model)" 
                                 % (d, dim))
        elif d == dim and n_al==n_alphas:
            indices = indices.reshape(n_al, d)
        else:
            raise ValueError("indices dimension is different from the"
                                 "model dimension : %d (indices) != %d (model)" 
                                 % (d, dim))
    return indices


def _check_margins(margins, dim):
    for marginal in margins:
        assert isinstance(marginal, ot.DistributionImplementation), \
            "The marginal should be an OpenTURNS implementation."
    assert len(margins) == dim, \
        "Incorrect dimension: %d!=%d" % (len(margins), dim)


def _check_copula(copula, dim):
    assert isinstance(copula, (ot.model_copula.Distribution,
                               ot.model_copula.CopulaImplementation,
                               ot.model_copula.DistributionImplementation)), \
        "The copula should be an OpenTURNS implementation: {}".format(
                                                                    type(copula))
    assert copula.getDimension() == dim, \
        "Incorrect dimension: %d!=%d" % (copula.getDimension(), dim)