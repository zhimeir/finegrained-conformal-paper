import numba as nb
import numpy as np
import openturns as ot

from .indices import BaseIndices, SensitivityResults_QOSE


def condMVN(mean, cov, dependent_ind, given_ind, X_given):
    """ 
    Returns conditional mean and variance of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!! ONLY FOR X MULTIVARIATE NORMAL !!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    
    cov = np.array(cov)
    
    B = cov[:, dependent_ind]
    B = B[dependent_ind]
    
    C = cov[:, dependent_ind]
    C = C[given_ind]
    
    D = cov[:, given_ind]
    D = D[given_ind]
    
    CDinv = np.dot(np.transpose(C), np.linalg.inv(D))
    
    condMean = mean[dependent_ind] + np.dot(CDinv, (X_given - mean[given_ind]))
    condVar = B - np.dot(CDinv, C)
    condVar = ot.CovarianceMatrix(condVar)
    
    return condMean, condVar

def _condMVN_new(cov, dependent_ind, given_ind, X_given):
    """
    Returns conditional mean and variance of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)
    """
    
    cov = np.asarray(cov)
    
    B = cov[:, dependent_ind]
    B = B[dependent_ind]
    
    C = cov[:, dependent_ind]
    C = C[given_ind]
    
    D = cov[:, given_ind]
    D = D[given_ind]
    
    CDinv = np.dot(np.transpose(C), np.linalg.inv(D))
    
    condMean = np.dot(CDinv, X_given)
    condVar = B - np.dot(CDinv, C)
    condVar = ot.CovarianceMatrix(condVar)
    
    return condMean, condVar

def cond_sampling(distribution, n_sample, idx, idx_c, x_cond):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!! ONLY FOR X MULTIVARIATE NORMAL !!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
    """
    cov = np.asarray(distribution.getCovariance())
    mean = np.asarray(distribution.getMean())
    cond_mean, cond_var = condMVN(mean, cov, idx, idx_c, x_cond)
    dist_cond = ot.Normal(cond_mean, cond_var)
    sample = dist_cond.getSample(n_sample)
    return sample

def _cond_sampling_new(distribution, n_sample, idx, idx_c, x_cond):
    """
    """
    margins_dep = [distribution.getMarginal(int(i)) for i in idx]
    margins_cond = [distribution.getMarginal(int(i)) for i in idx_c]

    # Creates a conditioned variables that follows a Normal distribution
    u_cond = np.zeros(x_cond.shape)
    for i, marginal in enumerate(margins_cond):
        u_cond[i] = np.asarray(ot.Normal().computeQuantile(marginal.computeCDF(x_cond[i])))

    sigma = np.asarray(distribution.getCopula().getCorrelation())
    cond_mean, cond_var = _condMVN_new(sigma, idx, idx_c, u_cond)
    
    n_dep = len(idx)
    dist_cond = ot.Normal(cond_mean, cond_var)
    sample_norm = np.asarray(dist_cond.getSample(int(n_sample)))
    sample_x = np.zeros((n_sample, n_dep))
    phi = lambda x: ot.Normal().computeCDF(x)
    for i in range(n_dep):
        u_i = np.asarray(phi(sample_norm[:, i].reshape(-1, 1))).ravel()
        sample_x[:, i] = np.asarray(margins_dep[i].computeQuantile(u_i)).ravel()

    return sample_x


def _sub_sampling(distribution, n_sample, idx):
    """
    Sampling from a subset of a given distribution.

    The function takes the margin and correlation matrix subset and creates a new copula
    and distribution function to sample.

    Parameters
    ----------


    Returns
    -------
    sample : array,
        The sample of the subset distribution.
    """

    # Margins of the subset
    margins_sub = [distribution.getMarginal(int(j)) for j in idx]
    # Get the correlation matrix
    sigma = np.asarray(distribution.getCopula().getCorrelation())
    # Takes only the subset of the correlation matrix
    copula_sub = ot.NormalCopula(ot.CorrelationMatrix(sigma[:, idx][idx, :]))
    # Creates the subset distribution
    dist_sub = ot.ComposedDistribution(margins_sub, copula_sub)
    # Sample
    sample = np.asarray(dist_sub.getSample(int(n_sample)))
    return sample


class QoseIndices(BaseIndices):
    """
    QOSE indices estimator.
    
    Estimates the QOSE indices for sensitivity analysis of model output. The
    estimation algorithm is inspired from [1] and slightly modified to 
    implement a bootstrap strategy. The bootstrap can be made on the random 
    permutation or the exact ones.
    
    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        An OpenTURNS distribution object.
    
    References
    ----------
    -- [1] Song, Eunhye, Barry L. Nelson, and Jeremy Staum, Shapley effects for
        global sensitivity analysis
        http://users.iems.northwestern.edu/~staum/ShapleyEffects.pdf
    """

    def __init__(self, input_distribution):
        BaseIndices.__init__(self, input_distribution)
        self._built_samples = False

    def build_sample(self, model, n_upsilon, n_perms, n_outer, n_inner):
        """
        Creates the input and output sample for the computation.
        
        Using Algorithm described in [1], the input sample are generated using
        the input distribution and are evaluated through the input model.
        
        Parameters
        ----------
        model : callable
            The input model function.
                    
        n_upsilon : int
            The sample size for the output averaged contrast function estimation.
        
        n_perms : int or None
            The number of permutations. If None, the exact permutation method
            is considerd.

        n_outer : int
            The number of conditionnal averaged contrast function estimations.
            
        n_inner : int
            The sample size for the conditionnal output contrast function estimation.
        
        References
        ----------
        -- [1] Song, Eunhye, Barry L. Nelson, and Jeremy Staum, Shapley effects for
            global sensitivity analysis
            http://users.iems.northwestern.edu/~staum/ShapleyEffects.pdf
        """

        assert callable(model), "The model function should be callable."
        assert isinstance(n_perms, (int, type(None))), \
            "The number of permutation should be an integer or None."
        assert isinstance(n_upsilon, int), "n_upsilon should be an integer."
        assert isinstance(n_outer, int), "n_outer should be an integer."
        assert isinstance(n_inner, int), "n_inner should be an integer."
        if isinstance(n_perms, int):
            assert n_perms > 0, "The number of permutation should be positive"
            
        assert n_upsilon > 0, "n_upsilon should be positive"
        assert n_outer > 0, "n_outer should be positive"
        assert n_inner > 0, "n_inner should be positive"
        
        dim = self.dim
        
        # All permutations used to compute the index
        if n_perms is None:
            estimation_method = 'exact'
            perms = list(ot.KPermutations(dim, dim).generate())
            n_perms = len(perms)
        else:
            estimation_method = 'random'
            perms = [np.random.permutation(dim) for i in range(n_perms)]
        
        # Creation of the design matrix
        input_sample_1 = np.asarray(self.input_distribution.getSample(n_upsilon), dtype=np.float64)
        input_sample_2 = np.empty((n_perms * (dim - 1) * n_outer * n_inner, dim), dtype=np.float64)

        for i_p, perm in enumerate(perms):
            idx_perm_sorted = np.argsort(perm)  # Sort the variable ids
            for j in range(dim - 1):
                # Normal set
                idx_j = perm[:j + 1]
                # Complementary set
                idx_j_c = perm[j + 1:]
                sample_j_c = _sub_sampling(self.input_distribution, n_outer, idx_j_c)
                self.sample_j_c = sample_j_c
                for l, xjc in enumerate(sample_j_c):
                    # Sampling of the set conditionally to the complementary
                    # element
                    xj = _cond_sampling_new(self.input_distribution, n_inner, idx_j, idx_j_c, xjc)
                    xx = np.c_[xj, [xjc] * n_inner]
                    ind_inner = i_p * (dim - 1) * n_outer * n_inner + j * n_outer * n_inner + l * n_inner
                    input_sample_2[ind_inner:ind_inner + n_inner, :] = xx[:, idx_perm_sorted]

        # Model evaluation
        input_sample = np.r_[input_sample_1, input_sample_2]
        output_sample = model(input_sample)

        self.input_sample = input_sample                
        self.output_sample_1 = output_sample[:n_upsilon]
        self.output_sample_2 = output_sample[n_upsilon:].reshape((n_perms, dim-1, n_outer, n_inner))
        
        self.model = model
        self.estimation_method = estimation_method
        self.n_upsilon = n_upsilon
        self.perms = perms
        self.n_outer = n_outer
        self.n_inner = n_inner
        self._built_samples = True

    def compute_indices(self, alpha, n_boot=1):
        """
        Computes the QOSE indices.
        
        The QOSE indices are computed from the computed samples.
        
        Parameters
        ----------
        alpha : array-like of shape = [n_alphas]
            The level of QOSE indices to assess.

        n_boot : int
            The number of bootstrap samples.
            
        Returns
        -------
        indice_results : instance of SensitivityResults
            The sensitivity results of the estimation.
        
        """
        assert self._built_samples, "The samples must be computed prior."
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            alpha = [alpha]
        assert isinstance(n_boot, int), "n_boot should be an integer."
        assert n_boot > 0, "n_boot should be positive."
        
        alpha = np.asarray(alpha, dtype=np.float64)
        dim = self.dim
        estimation_method = self.estimation_method
        perms = np.asarray(self.perms, dtype=np.uint32)
        n_alphas = alpha.shape[0]
        n_upsilon = self.n_upsilon
        n_perms = len(perms)
        n_outer = self.n_outer
        
        # Initialize QOSE indices for all players
        qose_indices = np.empty((dim, n_alphas, n_boot), dtype=np.float64)

        if estimation_method == 'exact':
            for i in range(n_boot):
                # Bootstrap sample indexes
                # The first iteration is computed over the whole sample.
                if i > 0:
                    boot_upsilon_idx = np.random.randint(0, n_upsilon, size=(n_upsilon, ), dtype=np.uint32)
                    boot_No_idx = np.random.randint(0, n_outer, size=(n_outer, ), dtype=np.uint32)
                else:
                    boot_upsilon_idx = np.arange(n_upsilon, dtype=np.uint32)
                    boot_No_idx = np.arange(n_outer, dtype=np.uint32)

                qose_indices[:,:,i] = _compute_qose_indices_exact(alpha,
                                                                  perms,
                                                                  self.output_sample_1,
                                                                  self.output_sample_2,
                                                                  boot_upsilon_idx,
                                                                  boot_No_idx)
                qose_indices_SE = None
        else:
            for i in range(n_boot):
                # Bootstrap sample indexes
                # The first iteration is computed over the whole sample.
                if i > 0:
                    boot_upsilon_idx = np.random.randint(0, n_upsilon, size=(n_upsilon, ), dtype=np.uint32)
                    boot_n_perms_idx = np.random.randint(0, n_perms, size=(n_perms, ), dtype=np.uint32)
                    qose_indices[:,:,i], _ = _compute_qose_indices_random(i,
                                                                          alpha,
                                                                          perms,
                                                                          self.output_sample_1,
                                                                          self.output_sample_2,
                                                                          boot_upsilon_idx,
                                                                          boot_n_perms_idx)
                else:
                    boot_upsilon_idx = np.arange(n_upsilon, dtype=np.uint32)
                    boot_n_perms_idx = np.arange(n_perms, dtype=np.uint32)
                    qose_indices[:,:,i], qose_indices_SE = _compute_qose_indices_random(i,
                                                                                        alpha,
                                                                                        perms,
                                                                                        self.output_sample_1,
                                                                                        self.output_sample_2,
                                                                                        boot_upsilon_idx,
                                                                                        boot_n_perms_idx)

        if hasattr(self.model, 'name'):
            if self.model.name in ('Additive Gaussian', 'Exponential Gaussian'):
                true_qose_indices = self.model.qosa_indices[2]
            else:
                true_qose_indices = None
        else:
            true_qose_indices = None

        indice_results = SensitivityResults_QOSE(alpha=alpha,
                                                 qose_indices=qose_indices,
                                                 true_qose_indices=true_qose_indices,
                                                 qose_indices_SE=qose_indices_SE,
                                                 estimation_method=estimation_method)
        return indice_results




# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------
#
# Private ancillary functions for the previous classes
#
# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------

# ----------------------------------------------------------------------------- 
# Numba functions to compute the averaged quantile contrast function 
# -----------------------------------------------------------------------------

@nb.njit("float64[:](float64[:], float64[:])", nogil=True, cache=True, parallel=True)
def _averaged_check_function_alpha_array_parallel(Y, alpha):
    """
    Definition of the check function also called pinball loss function.
    """

    n_alphas = alpha.shape[0]
    n_samples = Y.shape[0]
    qY = np.quantile(Y, alpha)

    check_function = np.empty((n_alphas, n_samples), dtype=np.float64)
    for i in nb.prange(n_samples):
        for j in range(n_alphas):
            u = Y[i] - qY[j]
            check_function[j,i] = u*(alpha[j] - (u < 0.))

    averaged_check_function = np.empty((n_alphas), dtype=np.float64)
    for i in nb.prange(n_alphas):
        averaged_check_function[i] = check_function[i,:].mean()

    return averaged_check_function


@nb.njit("float64[:](float64[:], float64[:])", nogil=True, cache=True, parallel=False)
def _averaged_check_function_alpha_array_unparallel(Y, alpha):
    """
    Definition of the check function also called pinball loss function.
    """

    n_alphas = alpha.shape[0]
    n_samples = Y.shape[0]
    qY = np.quantile(Y, alpha)

    check_function = np.empty((n_alphas, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_alphas):
            u = Y[i] - qY[j]
            check_function[j,i] = u*(alpha[j] - (u < 0.))

    averaged_check_function = np.empty((n_alphas), dtype=np.float64)
    for i in range(n_alphas):
        averaged_check_function[i] = check_function[i,:].mean()

    return averaged_check_function


# ----------------------------------------------------------------------------- 
# Numba functions to estimate the Qose indices
# -----------------------------------------------------------------------------

@nb.njit("float64[:,:](float64[:], uint32[:,:], float64[:], float64[:,:,:,:], uint32[:], uint32[:])", nogil=True, cache=True, parallel=True)
def _compute_qose_indices_exact(alpha, perms, output_sample_1, output_sample_2, boot_upsilon_idx, boot_No_idx):
    """
    Compute the QOSE indices for a given bootstrap sample.
    """
    
    n_alphas = alpha.shape[0]
    n_perms, dim, n_outer, n_inner = output_sample_2.shape
    dim += 1
    
    # Averaged contrast function
    upsilon_y =  _averaged_check_function_alpha_array_parallel(output_sample_1[boot_upsilon_idx],
                                                               alpha)
    
    # Compute the conditional averaged contrast functions as well as their expectations
    c_mean_upsilon = np.empty((n_perms, dim-1, n_alphas), dtype=np.float64)
    output_sample_2_temp = output_sample_2[:, :, boot_No_idx]
    for i in nb.prange(n_perms):
        c_upsilon = np.empty((n_alphas, n_outer), dtype=np.float64) # each proc/thread will have its own copy (i.e. private copy)
        for j in range(dim-1):
            for k in range(n_outer):
                c_upsilon[:,k] = _averaged_check_function_alpha_array_unparallel(output_sample_2_temp[i,j,k,:], alpha)
            for l in range(n_alphas):
                c_mean_upsilon[i,j,l] = c_upsilon[l,:].mean()
    
    # Cost variation
    delta_c = np.empty((n_perms, dim, n_alphas), dtype=np.float64)
    for i in nb.prange(n_perms):
        for j in range (dim):
            if j == 0:
                for k in range(n_alphas):
                    delta_c[i,j,k] = c_mean_upsilon[i,j,k]
            elif j == (dim-1):
                for k in range(n_alphas):
                    delta_c[i,j,k] = upsilon_y[k] - c_mean_upsilon[i,j-1,k]
            else:
                for k in range(n_alphas):
                    delta_c[i,j,k] = c_mean_upsilon[i,j,k] - c_mean_upsilon[i,j-1,k]
    
    # Compute the QOSE indices by aggregating the various costs computed above
    qose_indices = np.zeros((dim, n_alphas), dtype=np.float64)
    for i_p, perm in enumerate(perms):
        qose_indices[perm, :] += delta_c[i_p, :, :]
    
    # Loop necessary because broadcasting not working when "parallel=True", 
    # otherwise with parallel=False only qose_indices /= upsilon_y
    for i in nb.prange(dim):
        qose_indices[i,:] /= upsilon_y
    qose_indices /= n_perms

    return qose_indices


@nb.njit("UniTuple(float64[:,:], 2)(int32, float64[:], uint32[:,:], float64[:], float64[:,:,:,:], uint32[:], uint32[:])", nogil=True, cache=True, parallel=True)
def _compute_qose_indices_random(i_boot, alpha, perms, output_sample_1, output_sample_2, boot_upsilon_idx, boot_n_perms_idx):
    """
    Compute the QOSE indices for a given bootstrap sample.
    """
    
    n_alphas = alpha.shape[0]
    n_perms, dim, n_outer, n_inner = output_sample_2.shape
    dim += 1
    
    # Averaged contrast function
    upsilon_y =  _averaged_check_function_alpha_array_parallel(output_sample_1[boot_upsilon_idx],
                                                               alpha)
    
    # Compute the conditional averaged contrast functions as well as their expectations
    c_mean_upsilon = np.empty((n_perms, dim-1, n_alphas), dtype=np.float64)
    output_sample_2_temp = output_sample_2[boot_n_perms_idx]
    for i in nb.prange(n_perms):
        c_upsilon = np.empty((n_alphas, n_outer), dtype=np.float64) # each proc/thread will have its own copy (i.e. private copy)
        for j in range(dim-1):
            for k in range(n_outer):
                c_upsilon[:,k] = _averaged_check_function_alpha_array_unparallel(output_sample_2_temp[i,j,k,:], alpha)
            for l in range(n_alphas):
                c_mean_upsilon[i,j,l] = c_upsilon[l,:].mean()
    
    # Cost variation
    delta_c = np.empty((n_perms, dim, n_alphas), dtype=np.float64)
    for i in nb.prange(n_perms):
        for j in range (dim):
            if j == 0:
                for k in range(n_alphas):
                    delta_c[i,j,k] = c_mean_upsilon[i,j,k]
            elif j == (dim-1):
                for k in range(n_alphas):
                    delta_c[i,j,k] = upsilon_y[k] - c_mean_upsilon[i,j-1,k]
            else:
                for k in range(n_alphas):
                    delta_c[i,j,k] = c_mean_upsilon[i,j,k] - c_mean_upsilon[i,j-1,k]
    
    # Compute the QOSE indices by aggregating the various costs computed above
    qose_indices = np.zeros((dim, n_alphas), dtype=np.float64)
    qose_indices_2 = np.zeros((dim, n_alphas), dtype=np.float64)
    for i_p, perm in enumerate(perms[boot_n_perms_idx]):
        qose_indices[perm, :] += delta_c[i_p, :, :]
        
        # Estimate E[X**2] where X = marginal contribution of the player i
        if i_boot == 0:
            qose_indices_2[perm, :] += delta_c[i_p, :, :]**2
    
    # Loop necessary because broadcasting not working when "parallel=True", 
    # otherwise with parallel=False only qose_indices /= upsilon_y
    for i in nb.prange(dim):
        qose_indices[i,:] /= upsilon_y
        
        if i_boot == 0:
            qose_indices_2[i,:] /= upsilon_y**2
    qose_indices /= n_perms        
    qose_indices_2 /= n_perms
    
    # Compute the asymptotic standard deviation (i.e. sigma/sqrt(m)) to compute confidence intervals with CLT
    # for qose indices when using the random permutation method
    if i_boot == 0:
        qose_indices_SE = np.sqrt((qose_indices_2 - qose_indices**2)/n_perms)
    else:
        qose_indices_SE = np.empty((1,1), dtype=np.float64)
    
    return qose_indices, qose_indices_SE