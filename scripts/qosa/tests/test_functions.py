# -*- coding: utf-8 -*-

"""
Add docstring of the module
"""


import numpy as np
import openturns as ot
from scipy.stats import norm as ss_norm


from ._additive_exponential import compute_qosa_indices_additive_exponential
from ._toy_insurance import compute_qosa_indices_toy_insurance
from qosa.model import ProbabilisticModel


__all__ = ['AdditiveExponential', 'AdditiveGaussian', 'BiAdditiveLogNormal',
           'BiDifferenceExponential', 'ExponentialGaussian', 'ToyInsurance']


def is_independent(dist):
    """
    Check if the distribution has independent inputs.

    Parameters
    ----------
    dist : ot.Distribution,
        An multivariate OpenTURNS distribution object.

    Returns
    ------
    is_ind : bool,
        True if the distribution is independent, False otherwise.
    """

    is_ind = np.all(np.tril(np.asarray(dist.getCorrelation()), k=-1) == 0.)
    return is_ind


def ishigami_variance(a, b):
    return 0.5 + a**2/8 + b**2*np.pi**8/18 + b * np.pi**4/5


def ishigami_partial_variance(a, b):
    v1 = 0.5 * (1 + b * np.pi**4 / 5)**2
    v2 = a**2 / 8
    v3 = 0
    return v1, v2, v3


def ishigami_total_variance(a, b):
    v13 = b**2 * np.pi**8 * 8 / 225
    v1, v2, v3 = ishigami_partial_variance(a, b)
    return v1 + v13, v2, v3 + v13


class Ishigami(ProbabilisticModel):
    """
    This class collect all the information about the Ishigami test function
    for sensitivity analysis.
    """

    def __init__(self, a=7., b=0.1):
        dim = 3
        margins = [ot.Uniform(-np.pi, np.pi)]*dim
        copula = ot.IndependentCopula(dim)
        ProbabilisticModel.__init__(self,
                                    model_func=ishigami_func,
                                    input_distribution=ot.ComposedDistribution(margins, copula))
        self.a = a
        self.b = b
        self.name = 'Ishigami'

    @property
    def first_sobol_indices(self):
        """
        """
        if is_independent(self._input_distribution):
            a, b = self.a, self.b
            var_y = ishigami_variance(a, b)
            partial_var = ishigami_partial_variance(a, b)
            si = [vi / var_y for vi in partial_var]
            return np.asarray(si)
        else:
            return None

    @property
    def total_sobol_indices(self):
        """
        """
        if is_independent(self._input_distribution):
            a, b = self.a, self.b
            var_y = ishigami_variance(a, b)
            total_var = ishigami_total_variance(a, b)
            si = [vi / var_y for vi in total_var]
            return np.asarray(si)
        else:
            return None

    @property
    def shapley_indices(self):
        """
        """
        if is_independent(self._input_distribution):
            return np.asarray([0.437, 0.441, 0.12])
        else:
            return None

    @property
    def output_variance(self):
        a, b = self.a, self.b
        var_y = ishigami_variance(a, b)
        return var_y


class NegativeExponential(ot.PythonDistribution):
    """
    Class allowing to generate a negative exponential distribution.
    """

    def __init__(self, lambda_param=1):
        super(NegativeExponential, self).__init__(1)
        self.lambda_param = lambda_param
        self.exponential_dist = ot.Exponential(lambda_param)

    def getRealization(self):
        pt = self.exponential_dist.getRealization()
        pt[0] = -pt[0]
        return pt
    
    def getSample(self, size):
        return -np.array(self.exponential_dist.getSample(size))
    
    def computeCDF(self, X):
        CDF = 1 - np.array(self.exponential_dist.computeCDF(-np.array(X).reshape(-1,1)))
        return CDF


class AdditiveExponential(ProbabilisticModel):
    """
    This class collects all the information about the sum of Exponential
    distribution which gives a Hypoexponential distribution
    """

    def __init__(self, dim=4, lambda_params=None):
        if lambda_params is None:
            lambda_params = np.asarray([0.5, 0.65, 0.8, 0.95])
        else:
            lambda_params = np.asarray(lambda_params)
        n_lambda_params = lambda_params.size

        if dim != n_lambda_params:
            raise ValueError("lambda_parameters dimension is different from the"
                             "model dimension : %d (lambda_params) != %d (model)" 
                             % (n_lambda_params, dim))

        margins = [ot.Exponential(lambda_params[i]) for i in range(n_lambda_params)]
        copula = ot.IndependentCopula(dim)
        input_distribution = ot.ComposedDistribution(margins, copula)

        super(AdditiveExponential, self).__init__(
                                        model_func=additive_func,
                                        input_distribution=input_distribution)
        self.lambda_params = lambda_params
        self.name = 'Additive_Exponential'

    @ProbabilisticModel.qosa_indices.getter
    def qosa_indices(self):
        """
        Method to compute the Qosa indices for this model function of
        the alpha order.
        """

        if self.alpha is None:
            raise AttributeError("You should specify the alpha attribute in"
                                 " order to know at which alpha order matches"
                                 " the qosa indices  that you will get.")
        else:
            alpha = self.alpha

        if is_independent(self._input_distribution):
            qosa_indices = compute_qosa_indices_additive_exponential(self.lambda_params, 
                                                                     alpha)
            return qosa_indices
        else:
            return None


class AdditiveGaussian(ProbabilisticModel):
    """
    This class collects all the information about the Additive Gaussian test 
    function for sensitivity analysis.
    """

    def __init__(self, dim=2, means=None, std=None, beta=None):

        if means is None:
            means = np.zeros((dim, ))
        if std is None:
            std = np.ones((dim, ))

        n_means = np.asarray(means).shape[0]
        n_std = np.asarray(std).shape[0]
        assert n_means == n_std, "means and std have inconsistent dimension. dim means %d, dim std %d" % (n_means, n_std)
        if dim != n_means:
            raise ValueError("parameters dimension is different of the"
                             " model dimension : %d (params) != %d (model)" 
                             % (n_means, dim))

        margins = [ot.Normal(means[i], std[i]) for i in range(dim)]
        copula = ot.NormalCopula(dim)
        input_distribution = ot.ComposedDistribution(margins, copula)

        super(AdditiveGaussian, self).__init__(
                                            model_func=additive_func,
                                            input_distribution=input_distribution)
        self.beta = beta
        self.name = 'Additive Gaussian'

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta is None:
            beta = np.ones((self.dim, ))
        else:
            beta = np.asarray(beta)

        self.model_func = lambda x: additive_func(x, beta)
        self._beta = beta

    @property   
    def output_variance(self):
        covariance = np.asarray(self.input_distribution.getCovariance())
        var_y = (self.beta.dot(covariance)).dot(self.beta)
        return var_y

    @ProbabilisticModel.qosa_indices.getter
    def qosa_indices(self):
        """
        Method to compute the Qosa indices for this model function of
        the alpha order.
        """

        if self.alpha is None:
            raise AttributeError("You should specify the alpha attribute in"
                                 " order to know at which alpha order matches"
                                 " the qosa indices  that you will get.")
        else:
            alpha = self.alpha

        var_y = self.output_variance
        beta = self.beta
        dim = self.dim
        sigma = np.asarray(self.input_distribution.getCovariance())
        input_variance = sigma.diagonal()
        first_order_qosa_indices = np.empty((alpha.shape[0], dim), dtype=np.float64)
        total_order_qosa_indices = np.empty_like(first_order_qosa_indices, dtype=np.float64)
        
        for i in range(dim):
            c_i = np.asarray([j for j in range(dim) if j != i])

            # compute the first order qosa index
            inv_i = 1./input_variance[i]
            tmp_i = sigma[c_i, :][:, c_i] - inv_i*(np.outer(sigma[c_i, i], sigma[i, c_i]))
            var_w_i_given_i = np.asarray(beta[c_i].dot(tmp_i)).dot(beta[c_i])
            first_order_qosa_indices[:, i] = 1 - np.sqrt(var_w_i_given_i)/np.sqrt(var_y)

            # compute the total order qosa index
            inv_i = np.linalg.inv(sigma[c_i, :][:, c_i])
            tmp_i = input_variance[i] - np.asarray(sigma[i, c_i].dot(inv_i)).dot(sigma[c_i, i])
            var_i_given_w_i = beta[i]**2*tmp_i
            total_order_qosa_indices[:, i] = np.sqrt(var_i_given_w_i)/np.sqrt(var_y)
        
        if dim == 2:
            # Quantile oriented Shapley effects in dimension d=2
            shapley_qosa_indices = np.empty_like(first_order_qosa_indices, dtype=np.float64)
            rho = np.asarray(self.copula.getParameter())[0]
            coeff_correl = np.sqrt(1 - rho**2)
            coeff_1 = (np.abs(beta[0])*np.sqrt(input_variance[0])*coeff_correl)/np.sqrt(var_y)
            coeff_2 = (np.abs(beta[1])*np.sqrt(input_variance[1])*coeff_correl)/np.sqrt(var_y)
            shapley_qosa_indices[:, 0] = 0.5 - 0.5*coeff_2 + 0.5*coeff_1
            shapley_qosa_indices[:, 1] = 0.5 - 0.5*coeff_1 + 0.5*coeff_2

            return first_order_qosa_indices, total_order_qosa_indices, shapley_qosa_indices
        elif dim == 3 and np.asarray(self.copula.getParameter())[:2].sum()==0:
            # Quantile oriented Shapley effects in dimension d=3
            shapley_qosa_indices = np.empty_like(first_order_qosa_indices, dtype=np.float64)
            rho = np.asarray(self.copula.getParameter())[2]
            coeff_correl = 1 - rho**2

            contrib_1 = np.abs(beta[0])*np.sqrt(input_variance[0])
            contrib_2 = np.abs(beta[1])*np.sqrt(input_variance[1])*np.sqrt(coeff_correl)
            contrib_3 = np.abs(beta[2])*np.sqrt(input_variance[2])*np.sqrt(coeff_correl)

            contrib_12 = np.sqrt(beta[0]**2*input_variance[0] + beta[1]**2*input_variance[1]*coeff_correl)
            contrib_13 = np.sqrt(beta[0]**2*input_variance[0] + beta[2]**2*input_variance[2]*coeff_correl)
            contrib_23 = np.sqrt(beta[1]**2*input_variance[1] + 2*rho*beta[1]*beta[2]*np.sqrt(input_variance[1])*np.sqrt(input_variance[2]) + beta[2]**2*input_variance[2])

            shapley_qosa_indices[:, 0] = (contrib_1 + 0.5*(contrib_12 - contrib_2) + 0.5*(contrib_13 - contrib_3) + (np.sqrt(var_y) - contrib_23))/(3*np.sqrt(var_y))
            shapley_qosa_indices[:, 1] = (contrib_2 + 0.5*(contrib_12 - contrib_1) + 0.5*(contrib_23 - contrib_3) + (np.sqrt(var_y) - contrib_13))/(3*np.sqrt(var_y))
            shapley_qosa_indices[:, 2] = (contrib_3 + 0.5*(contrib_13 - contrib_1) + 0.5*(contrib_23 - contrib_2) + (np.sqrt(var_y) - contrib_12))/(3*np.sqrt(var_y))

            return first_order_qosa_indices, total_order_qosa_indices, shapley_qosa_indices

        return first_order_qosa_indices, total_order_qosa_indices, None


class BiAdditiveLogNormal(ProbabilisticModel):
    """
    This class collects all the information about the Bi Additive Log Normal test 
    function for sensitivity analysis.
    """

    def __init__(self, means=None, std=None):
        dim = 2

        if means is None:
            means = np.zeros((dim, ))
        if std is None:
            std = np.ones((dim, ))

        n_means = np.asarray(means).shape[0]
        n_std = np.asarray(std).shape[0]
        assert n_means == n_std, "means and std have inconsistent dimension. dim means %d, dim std %d" % (n_means, n_std)
        if dim != n_means:
            raise ValueError("parameters dimension is different of the"
                             " model dimension : %d (params) != %d (model)" 
                             % (n_means, dim))

        margins = [ot.LogNormal(means[i], std[i]) for i in range(dim)]
        copula = ot.NormalCopula(dim)
        input_distribution = ot.ComposedDistribution(margins, copula)

        super(BiAdditiveLogNormal, self).__init__(
                                            model_func=additive_func,
                                            input_distribution=input_distribution)
        self.sigma_Log = std
        self.name = 'Bi Additive LogNormal'

    @ProbabilisticModel.qosa_indices.getter
    def qosa_indices(self):
        """
        Method to compute the Qosa indices for this model function of
        the alpha order.
        """

        if self.alpha is None:
            raise AttributeError("You should specify the alpha attribute in"
                                 " order to know at which alpha order matches"
                                 " the qosa indices that you will get.")
        else:
            alpha = self.alpha

        dim = self.dim
        means = np.asarray(self.input_distribution.getMean())
        sigma_Log = self.sigma_Log
        first_order_qosa_indices = np.empty((alpha.shape[0], dim), dtype=np.float64)
        rho = np.asarray(self.copula.getParameter())[0]
        coeff_correl = np.sqrt(1 - rho**2)

        # Compute the first order qosa indices
        num_1 = means[1]*(alpha - ss_norm.cdf(ss_norm.ppf(alpha) - sigma_Log[1]*coeff_correl))
        num_2 = means[0]*(alpha - ss_norm.cdf(ss_norm.ppf(alpha) - sigma_Log[0]*coeff_correl))

        Y = np.asarray(self.input_distribution.getSample(10**8)).sum(axis=1)
        alpha_quantile = np.quantile(Y, alpha)
        truncated_expectation = np.mean(Y.reshape(-1,1)*(Y.reshape(-1,1) <= alpha_quantile), axis=0)
        denum = alpha*means.sum() - truncated_expectation

        first_order_qosa_indices[:, 0] = 1 - num_1/denum
        first_order_qosa_indices[:, 1] = 1 - num_2/denum

        # Compute the qose indices
        shapley_qosa_indices = np.empty_like(first_order_qosa_indices, dtype=np.float64)
        coeff_1 = (alpha*means[1] + ss_norm.cdf(ss_norm.ppf(alpha) - sigma_Log[0]*coeff_correl)*means[0])/denum
        coeff_2 = (alpha*means[0] + ss_norm.cdf(ss_norm.ppf(alpha) - sigma_Log[1]*coeff_correl)*means[1])/denum
        shapley_qosa_indices[:, 0] = 0.5 + 0.5*coeff_2 - 0.5*coeff_1
        shapley_qosa_indices[:, 1] = 0.5 + 0.5*coeff_1 - 0.5*coeff_2

        return first_order_qosa_indices, shapley_qosa_indices


class BiDifferenceExponential(ProbabilisticModel):
    """
    This class collects all the information about the following model:
    Y = X_1 - X_2 with X_1,X_2 iid following Exp(lambda)
    """

    def __init__(self, lambda_param=1):
        dim = 2
        margins = [ot.Exponential(lambda_param),
                   ot.Exponential(lambda_param)]
        copula = ot.IndependentCopula(dim)
        input_distribution = ot.ComposedDistribution(margins, copula)

        super(BiDifferenceExponential, self).__init__(
                                        model_func=difference_expo,
                                        input_distribution=input_distribution)
        self.lambda_param = lambda_param
        self.name = 'Difference_Exponential'

    @ProbabilisticModel.qosa_indices.getter
    def qosa_indices(self):
        """
        Method to compute the Qosa indices for this model function of
        the alpha order.
        """

        if self.alpha is None:
            raise AttributeError("You should specify the alpha attribute in"
                                 " order to know at which alpha order matches"
                                 " the qosa indices that you will get.")
        else:
            alpha = self.alpha

        n_alphas = alpha.shape[0]
        dim = self.dim

        if is_independent(self._input_distribution) and self.lambda_param==1:
            qosa_indices = np.zeros((n_alphas, dim), dtype=np.float64)
            for i, alpha_temp in enumerate(alpha):
                if alpha_temp >= 0.5:
                    qosa_indices[i,0] = ((1-alpha_temp)*(1-np.log(2*(1-alpha_temp)))+
                                        alpha_temp*np.log(alpha_temp))/(
                                        (1-alpha_temp)*(1-np.log(2*(1-alpha_temp))))
                    qosa_indices[i,1] = ((1-alpha_temp)*(1-np.log(2*(1-alpha_temp)))+
                                        (1-alpha_temp)*np.log(1-alpha_temp))/(
                                        (1-alpha_temp)*(1-np.log(2*(1-alpha_temp))))
                else:
                    qosa_indices[i,0] = (alpha_temp*(1-np.log(2*alpha_temp))+
                                        alpha_temp*np.log(alpha_temp))/(
                                        alpha_temp*(1-np.log(2*alpha_temp)))
                    qosa_indices[i,1] = (alpha_temp*(1-np.log(2*alpha_temp))+
                                        (1-alpha_temp)*np.log(1-alpha_temp))/(
                                        alpha_temp*(1-np.log(2*alpha_temp)))
            return qosa_indices
        else:
            return None


class ExponentialGaussian(ProbabilisticModel):
    """
    This class collects all the information about the Additive Gaussian test 
    function for sensitivity analysis.
    """

    def __init__(self, dim=2, means=None, std=None, beta=None):

        if means is None:
            means = np.zeros((dim, ))
        if std is None:
            std = np.ones((dim, ))

        n_means = np.asarray(means).shape[0]
        n_std = np.asarray(std).shape[0]
        assert n_means == n_std, "means and std have inconsistent dimension. dim means %d, dim std %d" % (n_means, n_std)
        if dim != n_means:
            raise ValueError("parameters dimension is different of the"
                             " model dimension : %d (params) != %d (model)" 
                             % (n_means, dim))

        margins = [ot.Normal(means[i], std[i]) for i in range(dim)]
        copula = ot.NormalCopula(dim)
        input_distribution = ot.ComposedDistribution(margins, copula)

        super(ExponentialGaussian, self).__init__(
                                            model_func=expo_func,
                                            input_distribution=input_distribution)
        self.beta = beta
        self.name = 'Exponential Gaussian'

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta is None:
            beta = np.ones((self.dim, ))
        else:
            beta = np.asarray(beta)

        self.model_func = lambda x: expo_func(x, beta)
        self._beta = beta

    @property   
    def output_variance(self):
        covariance = np.asarray(self.input_distribution.getCovariance())
        var_sum_X = (self.beta.dot(covariance)).dot(self.beta)
        return var_sum_X

    @ProbabilisticModel.qosa_indices.getter
    def qosa_indices(self):
        """
        Method to compute the Qosa indices for this model function of
        the alpha order.
        """

        if self.alpha is None:
            raise AttributeError("You should specify the alpha attribute in"
                                 " order to know at which alpha order matches"
                                 " the qosa indices  that you will get.")
        else:
            alpha = self.alpha

        var_sum_X = self.output_variance
        beta = self.beta
        dim = self.dim
        sigma = np.asarray(self.input_distribution.getCovariance())
        input_variance = sigma.diagonal()
        first_order_qosa_indices = np.empty((alpha.shape[0], dim), dtype=np.float64)
        total_order_qosa_indices = np.empty_like(first_order_qosa_indices, dtype=np.float64)

        for i in range(dim):
            c_i = np.asarray([j for j in range(dim) if j != i])

            # compute the first order qosa index
            inv_i = 1./input_variance[i]
            tmp_i = sigma[c_i, :][:, c_i] - inv_i*(np.outer(sigma[c_i, i], sigma[i, c_i]))
            var_sum_X_w_i_given_i = np.asarray(beta[c_i].dot(tmp_i)).dot(beta[c_i])
            num = alpha - ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(var_sum_X_w_i_given_i))
            denum = alpha - ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(var_sum_X))
            first_order_qosa_indices[:, i] = 1 - num/denum

            # compute the total order qosa index
            inv_i = np.linalg.inv(sigma[c_i, :][:, c_i])
            tmp_i = input_variance[i] - np.asarray(sigma[i, c_i].dot(inv_i)).dot(sigma[c_i, i])
            var_sum_X_i_given_w_i = beta[i]**2*tmp_i
            num = alpha - ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(var_sum_X_i_given_w_i))
            denum = alpha - ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(var_sum_X))
            total_order_qosa_indices[:, i] = num/denum
                
        if dim == 2:
            # Quantile oriented Shapley effects in dimension d=2
            shapley_qosa_indices = np.empty_like(first_order_qosa_indices, dtype=np.float64)
            rho = np.asarray(self.copula.getParameter())[0]
            coeff_correl = np.sqrt(1 - rho**2)
            denum = alpha - ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(var_sum_X))
            coeff_1 = ss_norm.cdf(ss_norm.ppf(alpha) - np.abs(beta[0])*np.sqrt(input_variance[0])*coeff_correl)/denum
            coeff_2 = ss_norm.cdf(ss_norm.ppf(alpha) - np.abs(beta[1])*np.sqrt(input_variance[1])*coeff_correl)/denum
            shapley_qosa_indices[:, 0] = 0.5 + 0.5*coeff_2 - 0.5*coeff_1
            shapley_qosa_indices[:, 1] = 0.5 + 0.5*coeff_1 - 0.5*coeff_2

            return first_order_qosa_indices, total_order_qosa_indices, shapley_qosa_indices
        elif dim == 3 and np.asarray(self.copula.getParameter())[:2].sum()==0:
            # Quantile oriented Shapley effects in dimension d=3
            shapley_qosa_indices = np.empty_like(first_order_qosa_indices, dtype=np.float64)
            rho = np.asarray(self.copula.getParameter())[2]
            coeff_correl = 1 - rho**2
            denum = alpha - ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(var_sum_X))

            contrib_1 = ss_norm.cdf(ss_norm.ppf(alpha) - np.abs(beta[0])*np.sqrt(input_variance[0]))
            contrib_2 = ss_norm.cdf(ss_norm.ppf(alpha) - np.abs(beta[1])*np.sqrt(input_variance[1])*np.sqrt(coeff_correl))
            contrib_3 = ss_norm.cdf(ss_norm.ppf(alpha) - np.abs(beta[2])*np.sqrt(input_variance[2])*np.sqrt(coeff_correl))

            contrib_12 = ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(beta[0]**2*input_variance[0] + beta[1]**2*input_variance[1]*coeff_correl))
            contrib_13 = ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(beta[0]**2*input_variance[0] + beta[2]**2*input_variance[2]*coeff_correl))
            contrib_23 = ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(beta[1]**2*input_variance[1] + 
                2*rho*beta[1]*beta[2]*np.sqrt(input_variance[1])*np.sqrt(input_variance[2]) + beta[2]**2*input_variance[2]))

            shapley_qosa_indices[:, 0] = ((alpha - contrib_1) 
                                          + 0.5*(contrib_2 - contrib_12) 
                                          + 0.5*(contrib_3 - contrib_13) 
                                          + (contrib_23 - ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(var_sum_X))))/(3*denum)
            shapley_qosa_indices[:, 1] = ((alpha - contrib_2) 
                                          + 0.5*(contrib_1 - contrib_12) 
                                          + 0.5*(contrib_3 - contrib_23) 
                                          + (contrib_13 - ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(var_sum_X))))/(3*denum)
            shapley_qosa_indices[:, 2] = ((alpha - contrib_3) 
                                          + 0.5*(contrib_1 - contrib_13) 
                                          + 0.5*(contrib_2 - contrib_23) 
                                          + (contrib_12 - ss_norm.cdf(ss_norm.ppf(alpha) - np.sqrt(var_sum_X))))/(3*denum)            
            return first_order_qosa_indices, total_order_qosa_indices, shapley_qosa_indices        

        return first_order_qosa_indices, total_order_qosa_indices, None


class ToyInsurance(ProbabilisticModel):
    """
    This class collects all the information about the sum of Exponential
    distribution
    """

    def __init__(self, GPD_params=[1.5, 0.25], LN_params=[1.1, 0.6, 0.], Gamma_params=[2., 0.6, 0.]):
        dim = 3
        margins = [ot.GeneralizedPareto(*GPD_params), ot.LogNormal(*LN_params), ot.Gamma(*Gamma_params)]
        copula = ot.IndependentCopula(dim)
        input_distribution = ot.ComposedDistribution(margins, copula)

        super(ToyInsurance, self).__init__(
                                        model_func=additive_func,
                                        input_distribution=input_distribution)
        self.GPD_params = GPD_params
        self.LN_params = LN_params
        self.Gamma_params = Gamma_params
        self.name = 'Toy_Insurance'

    @ProbabilisticModel.qosa_indices.getter
    def qosa_indices(self):
        """
        Method to compute the Qosa indices for this model function of
        the alpha order.
        """

        if self.alpha is None:
            raise AttributeError("You should specify the alpha attribute in"
                                 " order to know at which alpha order matches"
                                 " the qosa indices that you will get.")
        else:
            alpha = self.alpha

        if is_independent(self._input_distribution):
            GPD_params = self.GPD_params
            LN_params = self.LN_params
            Gamma_params = self.Gamma_params
            n_samples = 10**8
            qosa_indices = compute_qosa_indices_toy_insurance(GPD_params,
                                                              LN_params,
                                                              Gamma_params,
                                                              alpha,
                                                              n_samples)
            return qosa_indices
        else:
            return None


def additive_func(X, beta=None):
    """
    Additive function.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        The input variables.

    Returns
    -------
    y : float or array,
        The function output. If n > 1, the function returns an array.
    """
    
    x = np.asarray(X)
    if x.ndim == 1:
        dim = x.shape[0]
    else:
        n_sample, dim = x.shape

    if beta is None:
        beta = np.ones((dim, ))
    else:
        beta = np.asarray(beta)
    y = np.dot(x, beta)
    return y


def difference_expo(X):
    """
    Sum between a positive and negative exponential distribution.
    
    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        The input variables.
        
    Returns
    -------
    X.sum(axis=1) : array-like, shape = [n_samples]
        The function output.
    """

    dim = X.shape[1]
    assert dim == 2, "Dimension problem %d != %d " % (2, dim)

    return X[:,0] - X[:,1]


def expo_func(X, beta=None):
    """
    Additive function.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        The input variables.

    Returns
    -------
    y : float or array,
        The function output. If n > 1, the function returns an array.
    """
    
    x = np.asarray(X)
    if x.ndim == 1:
        dim = x.shape[0]
    else:
        n_sample, dim = x.shape

    if beta is None:
        beta = np.ones((dim, ))
    else:
        beta = np.asarray(beta)
    y = np.dot(x, beta)
    return np.exp(y)


def ishigami_func(x, a=7, b=0.1):
    """
    Ishigami function.

    Parameters
    ----------
    x : array,
        The input variables. The shape should be 3 x n.

    Returns
    -------
    y : float or array,
        The function output. If n > 1, the function returns an array.
    """

    x = np.asarray(x).squeeze()
    if x.shape[0] == x.size:
        dim = x.shape[0]
        y = np.sin(x[0]) + a*np.sin(x[1])**2 + b*x[2]**4 * np.sin(x[0])
    else:
        dim = x.shape[1]
        y = np.sin(x[:, 0]) + a*np.sin(x[:, 1])**2 + \
            b*x[:, 2]**4 * np.sin(x[:, 0])

    assert dim == 3, "Dimension problem %d != %d " % (3, ndim)

    return y