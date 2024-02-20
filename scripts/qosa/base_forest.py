# -*- coding: utf-8 -*-

"""
Add docstring of the module
"""


import numba as nb
import numpy as np
from numpy import float64 as DTYPE
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_X_y, check_random_state


__all__ = ['QuantileRegressionForest']


class BaseEstimatorForest(RandomForestRegressor):
    """
    Base class for all estimators using the Random Forest method in their
    operation.    

    This class builds a random forest using the Scikit-Learn package.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 n_estimators=100,
                 criterion="squared_error",
                 max_depth=None,
                 min_samples_split=40,
                 min_samples_leaf=20,
                 min_weight_fraction_leaf=0.,
                 max_features=0.5,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None):

        super(BaseEstimatorForest, self).__init__(
                            n_estimators=n_estimators,
                            criterion=criterion,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                            max_features=max_features,
                            max_leaf_nodes=max_leaf_nodes,
                            min_impurity_decrease=min_impurity_decrease,
                            bootstrap=bootstrap,
                            oob_score=oob_score,
                            n_jobs=n_jobs,
                            random_state=random_state,
                            verbose=verbose,
                            warm_start=warm_start,
                            ccp_alpha=ccp_alpha,
                            max_samples=max_samples)

    def fit(self, X, y):
        """
        Calibration of the forest with the input and output samples and 
        extraction of the leaf nodes for the data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            Input samples used to calibrate the forest.
        
        y : array-like, shape = [n_samples]
            Output samples used to calibrate the forest.

        Returns
        -------
        self : object
        """
        
        # Transformation of the data in np.array
        X = np.asarray(X, order='C')
        y = np.asarray(y)

        # Validate or convert the input data
        try:
            X, y = check_X_y(X, y, dtype=DTYPE, y_numeric=True)
        except ValueError as instance_ValueError:
            if X.shape[0] == X.size:
                X = X.reshape(-1, 1, order='C')
                X, y = check_X_y(X, y, dtype=DTYPE, y_numeric=True)
            else:
                raise ValueError(instance_ValueError)
        
        # Informations about the problem dimension
        self._n_samples, self._input_dim = X.shape

        # Fit the forest
        super(BaseEstimatorForest, self).fit(X, y)

        # Save the data. Necessary to compute the quantiles.
        self._input_samples = X
        self._output_samples = y

        # The resulting leaves of each element of the input sample.
        self._samples_nodes = self.apply(X)

        # Verify if the array is C-contiguous in memory in order to better 
        # browse it to increase the performance
        if self._samples_nodes.flags.c_contiguous == False:
            self._samples_nodes = np.ascontiguousarray(self._samples_nodes)
        
        return self

    def get_nodes(self, X):
        """
        Mainly for the classes QuantileRegressionForest and MinimumConditionalExpectedCheckFunctionForest.

        Function to get the leaf nodes within which an observation lives.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        ----------
        X_nodes : array-like of shape = [n_samples, n_estimators]
            For each datapoint x in X and for each tree in the forest, return 
            the index of the leaf where x ends up in.
        """
        
        X, _ = self._check_input(X)

        # Nodes of the regressor in all the trees
        # Shape : (numRegressor * numTree)
        X_nodes = self.apply(X)
        
        # Verify if the array is C-contiguous in memory in order to better 
        # browse it to increase the performance
        if X_nodes.flags.c_contiguous == False:
            X_nodes = np.ascontiguousarray(X_nodes)

        return X_nodes

    def _check_input(self, X):
        """
        Mainly for the classes QuantileRegressionForest and MinimumConditionalExpectedCheckFunctionForest.
        
        Function to verify the dimension of the data.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The elements where we want to assess the conditional quantiles.

        Returns
        -------
        X : array-like of shape = [n_samples, n_features]
            Converted array to get the data in the right format.

        n : int
            The number of points where to assess the conditional quantiles.
        """
        
        n = X.shape[0]  # Number of sample
        try:  # Works if X is a 2d array
            d = X.shape[1]  # Dimension of the array
            if d != self._input_dim:  # If the dimension is not correct
                if n == self._input_dim:  # There is one sample of d dimension
                    n = d # the number of samples
                    d = self._input_dim
                    X = np.ascontiguousarray(X.T)
                else:  # Error
                    raise ValueError("X dimension is different from forest "
                                     "dimension : %d (X) != %d (forest)" 
                                     % (d, self._input_dim))
        except:  # Its a vector
            d = 1
            if d != self._input_dim:  # If the dimension is not correct
                if n == self._input_dim:  # There is one sample of d dimension
                    d = n
                    n = 1
                    X = X.reshape(n, d, order='C')
                else:  # Error
                    raise ValueError("X dimension is different from forest "
                                     "dimension : %d (X) != %d (forest)" 
                                     % (d, self._input_dim))
            elif d == self._input_dim:
                X = X.reshape(n, 1, order='C')

        return X, n

    def _compute_inbag_samples(self):
        """
        Private function used to get the indices of the bootstrap samples of each 
        tree and the number of times that an observation is selected.
        """

        n_trees = self.n_estimators
        n_samples = self._n_samples        
        idx_bootstrap_samples = np.empty((n_samples, n_trees), dtype=np.uint32, order='F')
        inbag_samples = np.empty((n_samples, n_trees), dtype=np.uint32, order='F')
        
        for idx_tree in range(n_trees):
            idx_bootstrap_samples[:,idx_tree] = _generate_sample_indices(
                                                    self.estimators_[idx_tree].random_state,
                                                    n_samples,
                                                    n_samples)
            inbag_samples[:,idx_tree] = np.bincount(idx_bootstrap_samples[:,idx_tree],
                                                    minlength=n_samples)

        return idx_bootstrap_samples, inbag_samples

    def _compute_oob_samples(self):
        """
        Private function used to get the indices of the OOB (Out Of Bag) samples 
        of each tree.
        """

        n_trees = self.n_estimators
        n_samples = self._n_samples
        idx_oob_samples = nb.typed.Dict.empty(key_type=nb.types.int64, 
                                              value_type=nb.types.uint32[:])  
        
        # Here at each iteration we obtain out of bag samples for every tree.
        for idx_tree in range(n_trees):
            idx_oob_samples.update({idx_tree : _generate_unsampled_indices(
                                                    self.estimators_[idx_tree].random_state, 
                                                    n_samples,
                                                    n_samples)})

        return idx_oob_samples


class QuantileRegressionForest(BaseEstimatorForest):
    """
    Quantile Regresion Forest

    This class proposes several estimation methods based on the Random Forest 
    methodology to approximate the conditional cumulative distribution functions 
    as well as the conditional quantiles.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    
    n_jobs : int, default=-1
        The number of jobs to run in parallel for both fit and predict. If -1,
        then the number of jobs is set to the number of cores.

    For additional parameters, take a look to BaseEstimatorForest's parameters.
    """

    def fit(self, X, y, oob_score_quantile=False, alpha=None, method='Weighted_CDF', used_bootstrap_samples=False):
        """
        Calibration of the forest with the input and output samples.
        The last three parameters allow to estimate / assess the generalization 
        error of the estimation method called 'Averaged_Quantile' on the OOB samples
        at the alpha-level.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            Input samples used to calibrate the forest.
        
        y : array-like, shape = [n_samples]
            Output samples used to calibrate the forest.

        oob_score_quantile : bool, default=False
            Whether to use out-of-bag samples to estimate the generalization error
            for the estimation of the conditional quantiles.

        alpha : array-like of shape = [n_alphas], default=None
            The order where we want to assess the generalization error for the 
            conditional quantiles thanks to the OOB samples.

        method : str, default="Weighted_CDF"
            The estimation method used to compute the conditional quantiles of the OOB
            samples. Available estimation methods are: "Averaged_Quantile" or "Weighted_CDF".

        used_bootstrap_samples : bool, default=False
            Using the bootstrap samples or the original sample to compute the 
            conditional quantiles of the OOB observations.

        Returns
        -------
        self : object
        """

        if isinstance(alpha, (int, np.integer, float, np.floating)):
            alpha = [alpha]
        alpha = np.asarray(alpha)

        assert method in ("Averaged_Quantile", "Weighted_CDF"), "Given method is not known: %s" % (method,)

        assert isinstance(used_bootstrap_samples, bool), \
            "The parameter 'used_bootstrap_samples' should be a boolean."

        if not self.bootstrap and oob_score_quantile:
            raise ValueError("Out Of Bag estimation for quantiles only available"
                             " if bootstrap=True")
        
        if oob_score_quantile and None in alpha:
            raise ValueError("You need to specify the alpha order at which you want to"
                             " compute the Out Of Bag estimation if oob_score_quantile=True")

        if not oob_score_quantile and None not in alpha:
            raise ValueError("In addition to giving the alpha parameter, you need to specify"
                             " oob_score_quantile=True at the instanciation of your object"
                             " if you want to compute the Out Of Bag quantile error.")

        super(QuantileRegressionForest, self).fit(X, y)

        self.oob_score_quantile = oob_score_quantile
        if oob_score_quantile:
            alpha = alpha.astype(dtype=np.float64)
            alpha.sort()
            self._set_oob_score_quantile(method, alpha, used_bootstrap_samples)

        return self

    def predict_C_CDF(self, X, used_bootstrap_samples=False):
        """
        Compute the Conditional Cumulative Distribution Function(C_CDF) of the 
        elements X by using the Weighted approach developed in [1].
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The elements where we want to assess the C_CDF.
        
        used_bootstrap_samples : bool, default=False
            Using the bootstrap samples or the original sample to compute the 
            weights allowing to estimate the C_CDF.

        Returns
        -------
        C_CDF : array-like of shape = [n_samples, n_samples]
            The C_CDF computed at points X. Each row matches to a full estimation
            of the C_CDF based on the observations provided in the fit method, 
            i.e. the cumulative weighted sum given X=x.

        References
        ----------
        .. [1] Elie-Dit-Cosaque, Kévin, and Véronique Maume-Deschamps. "Random 
               forest estimation of conditional distribution functions and 
               conditional quantiles.", Preprint on HAL, 2020.
        """
        
        if isinstance(X, (int, np.integer, float, np.floating)):
            X = [X]

        assert isinstance(used_bootstrap_samples, bool), \
            "The parameter 'used_bootstrap_samples' should be a boolean."

        # Transformation to array for convenience
        X = np.asarray(X, order='C')

        # Leaf nodes within which the conditional observations fall into
        X_nodes = super(QuantileRegressionForest, self).get_nodes(X)

        # Selection of the unique elements on which computed the C_CDF
        _, idx_unique_X_nodes, idx_inverse_unique_X_nodes = np.unique(
                                                                X_nodes, 
                                                                axis=0,
                                                                return_index=True,
                                                                return_inverse=True)
        
        # Compute the C_CDF according the method chosen by the user
        C_CDF = np.empty((idx_unique_X_nodes.shape[0], self._output_samples.shape[0]),
                          dtype=np.float64,
                          order='C')
        if used_bootstrap_samples:
            _, inbag_samples = super(QuantileRegressionForest,
                                     self)._compute_inbag_samples()
            _compute_conditional_CDF_with_Weighted_approach(C_CDF,
                                                            self._output_samples,
                                                            self._samples_nodes,
                                                            inbag_samples,
                                                            X_nodes[idx_unique_X_nodes],
                                                            used_bootstrap_samples)
        else:
            inbag_samples = np.empty((1, 1), dtype=np.uint32)
            _compute_conditional_CDF_with_Weighted_approach(C_CDF,
                                                            self._output_samples,
                                                            self._samples_nodes,
                                                            inbag_samples,
                                                            X_nodes[idx_unique_X_nodes],
                                                            used_bootstrap_samples)
        return C_CDF[idx_inverse_unique_X_nodes]

    def predict_quantile(self, X, alpha, method='Weighted_CDF', used_bootstrap_samples=False):
        """
        Compute the conditional quantiles at the alpha level of the elements X by using 
        one of the two following methods: 
            - "Averaged_Quantile", approach developped in [1]
            - "Weighted_CDF", approach developped in [2]
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The elements where we want to assess the conditional quantiles.
        
        alpha : array-like of shape = [n_alphas]
            The level of the conditional quantiles to assess.

        method : str, default="Weighted_CDF"
            The estimation method used to compute the conditional quantiles.
            Available estimation methods are: "Averaged_Quantile" or "Weighted_CDF".

        used_bootstrap_samples : bool, default=False
            Using the bootstrap samples or the original sample to compute the 
            conditional quantiles.

        Returns
        -------
        quantiles : array-like of shape = [n_samples, n_alphas]
            The conditional quantiles computed at points X for the different
            values of alpha. If the alpha levels are provided in no particular order,
            e.g. alpha=[0.5, 0.1, 0.9], the returned values for each point / row 
            are sorted according to alpha levels, i.e. [0.1, 0.5, 0.9].

        References
        ----------
        .. [1] add reference
        .. [2] Elie-Dit-Cosaque, Kévin, and Véronique Maume-Deschamps. "Random 
               forest estimation of conditional distribution functions and 
               conditional quantiles.", Preprint on HAL, 2020.
        """
        
        if isinstance(alpha, (int, np.integer, float, np.floating)):
            alpha = [alpha]
        if isinstance(X, (int, np.integer, float, np.floating)):
            X = [X]

        assert isinstance(used_bootstrap_samples, bool), \
            "The parameter 'used_bootstrap_samples' should be a boolean."

        # Transformation to array for convenience
        alpha = np.asarray(alpha).astype(dtype=np.float64)
        alpha.sort()
        X = np.asarray(X, order='C')

        n_alphas = alpha.size  # Number of probabilities
        n_trees = self.n_estimators # Number of trees

        # Leaf nodes within which the conditional observations fall into
        X_nodes = super(QuantileRegressionForest, self).get_nodes(X)

        # Selection of the unique elements on which computed the conditional quantiles
        _, idx_unique_X_nodes, idx_inverse_unique_X_nodes = np.unique(
                                                                X_nodes, 
                                                                axis=0,
                                                                return_index=True,
                                                                return_inverse=True)
        
        # Compute the conditional quantiles according the method chosen by the user
        if method == 'Averaged_Quantile':
            quantiles = np.empty((n_trees, idx_unique_X_nodes.shape[0], n_alphas), 
                                 dtype=np.float64,
                                 order='C')
            if used_bootstrap_samples:
                idx_bootstrap_samples, _ = super(QuantileRegressionForest, 
                                                 self)._compute_inbag_samples()
                _compute_conditional_quantile_on_each_tree(quantiles, 
                                                           self._output_samples,
                                                           self._samples_nodes,
                                                           idx_bootstrap_samples,
                                                           X_nodes[idx_unique_X_nodes],
                                                           alpha,
                                                           used_bootstrap_samples)
            else:
                idx_bootstrap_samples = np.empty((1, 1), dtype=np.uint32)
                _compute_conditional_quantile_on_each_tree(quantiles, 
                                                           self._output_samples,
                                                           self._samples_nodes,
                                                           idx_bootstrap_samples,
                                                           X_nodes[idx_unique_X_nodes],
                                                           alpha,
                                                           used_bootstrap_samples)
            return (quantiles.sum(axis=0)/n_trees)[idx_inverse_unique_X_nodes]
        elif method == 'Weighted_CDF':
            quantiles = np.empty((idx_unique_X_nodes.shape[0], n_alphas),
                                 dtype=np.float64,
                                 order='C')
            if used_bootstrap_samples:
                _, inbag_samples = super(QuantileRegressionForest,
                                         self)._compute_inbag_samples()
                _compute_conditional_quantile_with_Weighted_CDF(quantiles,
                                                                self._output_samples,
                                                                self._samples_nodes,
                                                                inbag_samples,
                                                                X_nodes[idx_unique_X_nodes],
                                                                alpha,
                                                                used_bootstrap_samples)
            else:
                inbag_samples = np.empty((1, 1), dtype=np.uint32)
                _compute_conditional_quantile_with_Weighted_CDF(quantiles,
                                                                self._output_samples,
                                                                self._samples_nodes,
                                                                inbag_samples,
                                                                X_nodes[idx_unique_X_nodes],
                                                                alpha,
                                                                used_bootstrap_samples)
            return quantiles[idx_inverse_unique_X_nodes]

    def _set_oob_score_quantile(self, method, alpha, used_bootstrap_samples):
        """
        Compute out-of-bag predictions and scores for quantiles at the alpha level.
        """

        if method == 'Averaged_Quantile':
            idx_oob_samples = super(QuantileRegressionForest, 
                                self)._compute_oob_samples()
            if used_bootstrap_samples:
                idx_bootstrap_samples, _ = super(QuantileRegressionForest,
                                             self)._compute_inbag_samples()
                conditional_quantiles =_compute_conditional_quantile_for_oob_samples_on_each_tree_with_bootstrap_data(
                                                                    self._output_samples,
                                                                    self._samples_nodes,
                                                                    idx_bootstrap_samples,
                                                                    idx_oob_samples,
                                                                    alpha)
            else:
                conditional_quantiles =_compute_conditional_quantile_for_oob_samples_on_each_tree_with_original_data(
                                                                    self._output_samples,
                                                                    self._samples_nodes,
                                                                    idx_oob_samples,
                                                                    alpha)
        elif method == 'Weighted_CDF':
            _, inbag_samples = super(QuantileRegressionForest,
                                     self)._compute_inbag_samples()
            conditional_quantiles = np.zeros((self._n_samples, alpha.shape[0]),
                                             dtype = np.float64,
                                             order = 'C')
            if used_bootstrap_samples:
                _compute_conditional_quantile_for_oob_samples_with_Weighted_CDF_using_bootstrap_data(
                                                                    conditional_quantiles,
                                                                    self._output_samples,
                                                                    self._samples_nodes,
                                                                    inbag_samples,
                                                                    alpha)
            else:
                _compute_conditional_quantile_for_oob_samples_with_Weighted_CDF_using_original_data(
                                                                    conditional_quantiles,
                                                                    self._output_samples,
                                                                    self._samples_nodes,
                                                                    inbag_samples,
                                                                    alpha)

        # if (conditional_quantiles.sum(axis=1) == 0).any():
        #    warn("Some inputs do not have OOB scores. This probably means too few trees were"
        #         " used to compute any reliable oob estimates.")

        self.oob_prediction_quantile_ = conditional_quantiles
        self.oob_score_quantile_ = _averaged_check_function_alpha_array(
                                    self._output_samples.reshape(-1,1) - conditional_quantiles,
                                    alpha)


class MinimumConditionalExpectedCheckFunctionWithLeaves(BaseEstimatorForest):
    """
    Minimum Conditional Expected Check Function With Leaves

    This class computes the expected value of the check function used in the 
    QOSA index. It first averages over all the leaves of a tree the minimum got 
    within each one, then over all the trees. 

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    
    n_jobs : int, default=-1
        The number of jobs to run in parallel for both fit and predict. If -1,
        then the number of jobs is set to the number of cores.

    For additional parameters, take a look to BaseEstimatorForest's parameters.
    """

    def predict(self, alpha, used_bootstrap_samples=False):
        """
        Compute the expected optimal value of the conditional contrast function.
        A detailed description can be found in [1].
        
        Parameters
        ----------      
        alpha : array-like of shape = [n_alphas]
            The order where we want to assess the expected check function.

        used_bootstrap_samples : bool, default=False
            Using the bootstrap samples or the original sample to compute the 
            minimum in each leaf.

        Returns
        -------
        expected_check_function : array-like of shape = [n_alphas, 2]
            The expected check function assessed at the alpha levels. First
            (resp. Second) column for the classical (resp .weighted) mean, i.e. 
            depending of the number of observations within each leaf. 
            If the alpha levels are provided in no particular order, e.g. 
            alpha=[0.5, 0.1, 0.9], the returned values in each column are sorted 
            according to alpha levels, i.e. [0.1, 0.5, 0.9].
        
        References
        ----------
        .. [1] add reference
        """

        if isinstance(alpha, (int, np.integer, float, np.floating)):
            alpha = [alpha]

        assert isinstance(used_bootstrap_samples, bool), \
            "The parameter 'used_bootstrap_samples' should be a boolean."

        # Transformation to array for convenience
        alpha = np.asarray(alpha).astype(dtype=np.float64)
        alpha.sort() # Very important to have alpha sorted for the numba function.
        
        if used_bootstrap_samples:
            idx_bootstrap_samples, _ = super(MinimumConditionalExpectedCheckFunctionWithLeaves, 
                                             self)._compute_inbag_samples()
            expected_check_function = _compute_expected_value_check_function(
                                                            self._output_samples,
                                                            self._samples_nodes,
                                                            idx_bootstrap_samples,
                                                            alpha,
                                                            used_bootstrap_samples)
        else:
            idx_bootstrap_samples = np.empty((1, 1), dtype=np.uint32)
            expected_check_function = _compute_expected_value_check_function(
                                                            self._output_samples,
                                                            self._samples_nodes,
                                                            idx_bootstrap_samples,
                                                            alpha,
                                                            used_bootstrap_samples)
        return expected_check_function


class MinimumConditionalExpectedCheckFunctionWithWeights(BaseEstimatorForest):
    """
    Minimum Conditional Expected Check Function With Weights

    This class computes the expected value of the check function used in the 
    QOSA index. The idea is to look for the minimum of a weighted estimate of 
    the conditional check function.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    
    n_jobs : int, default=-1
        The number of jobs to run in parallel for both fit and predict. If -1,
        then the number of jobs is set to the number of cores.

    For additional parameters, take a look to BaseEstimatorForest's parameters.
    """

    def predict(self, X, alpha, used_bootstrap_samples=False):
        """
        Compute the expected optimal value of the conditional contrast function
        for the elements X at the alpha level. A detailed description can be 
        found in [1].
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The elements where we want to assess the the conditional contrast function.
        
        alpha : array-like of shape = [n_alphas]
            The alpha order in which to evaluate the conditional contrast function.

        used_bootstrap_samples : bool, default=False
            Using the bootstrap samples or the original sample to compute the weights.

        Returns
        -------
        minimum_expectation : array-like of shape = [n_samples, n_alphas]
            The expected value of the conditional check function computed at points X for
            the different values of alpha. 
            If the alpha levels are provided in no particular order,
            e.g. alpha=[0.5, 0.1, 0.9], the returned values for each point / row 
            are sorted according to alpha levels, i.e. [0.1, 0.5, 0.9].
        
        References
        ----------
        .. [1] add reference
        """

        if isinstance(alpha, (int, np.integer, float, np.floating)):
            alpha = [alpha]
        if isinstance(X, (int, np.integer, float, np.floating)):
            X = [X]

        assert isinstance(used_bootstrap_samples, bool), \
            "The parameter 'used_bootstrap_samples' should be a boolean."

        # Transformation to array for convenience
        alpha = np.asarray(alpha).astype(dtype=np.float64)
        alpha.sort() # Very important to have alpha sorted for the numba function.
        X = np.asarray(X, order='C')
        
        n_alphas = alpha.size  # Number of probabilities

        # Leaf nodes within which the conditional observations fall into
        X_nodes = super(MinimumConditionalExpectedCheckFunctionWithWeights, self).get_nodes(X)

        # Selection of the unique elements on which computed the conditional
        # contrast function
        _, idx_unique_X_nodes, idx_inverse_unique_X_nodes = np.unique(
                                                                X_nodes, 
                                                                axis=0,
                                                                return_index=True,
                                                                return_inverse=True)

        minimum_expectation = np.empty((idx_unique_X_nodes.shape[0], n_alphas), 
                                       dtype=np.float64,
                                       order='C')
        if used_bootstrap_samples:
            _, inbag_samples = super(MinimumConditionalExpectedCheckFunctionWithWeights,
                                     self)._compute_inbag_samples()
            _compute_conditional_minimum_expectation(minimum_expectation,
                                                     self._output_samples,
                                                     self._samples_nodes,
                                                     inbag_samples,
                                                     X_nodes[idx_unique_X_nodes],
                                                     alpha,
                                                     used_bootstrap_samples)
        else:
            inbag_samples = np.empty((1, 1), dtype=np.uint32)
            _compute_conditional_minimum_expectation(minimum_expectation,
                                                     self._output_samples,
                                                     self._samples_nodes,
                                                     inbag_samples,
                                                     X_nodes[idx_unique_X_nodes],
                                                     alpha,
                                                     used_bootstrap_samples)

        return minimum_expectation[idx_inverse_unique_X_nodes]




# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------
#
# Private ancillary functions for the previous classes
#
# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------

# ---------------------------------
# For the class BaseEstimatorForest
# ---------------------------------

def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function available in the sklearn.ensemble.forest module until now (01/2020) 
    but it is deprecated in version 0.22 and will be removed in version 0.24.
    ==> copy-paste of the method in order to use it

    Function giving the bootstrap samples used to build a tree
    """

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, 
                                             n_samples,
                                             n_samples_bootstrap,
                                             dtype=np.uint32)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function available in the sklearn.ensemble.forest module until now (01/2020)
    but it is deprecated in version 0.22 and will be removed in version 0.24.
    ==> copy-paste of the method in order to use it

    Function giving the Out Of Bag samples of a tree
    """

    sample_indices = _generate_sample_indices(random_state, n_samples,
                                              n_samples_bootstrap)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples, dtype=np.uint32)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


# -----------------------------------------------------------------------------------------------
# For the classes MinimumConditionalExpectedCheckFunctionWithWeights and QuantileRegressionForest
# -----------------------------------------------------------------------------------------------

@nb.njit("float64[:](int64[:,:], uint32[:,:], int64[:])", nogil=True, cache=True, parallel=False)
def _compute_weight_with_bootstrap_data(samples_nodes, inbag_samples, X_nodes_k):
    """
    Function to compute the bootstrapped averaged weight of each individual of
    the original training sample in the forest.
    """
    
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_elements = samples_nodes.size
    
    col_cnt = np.zeros((n_trees), dtype=np.uint32)
    row_idx = np.empty((n_samples + 1), dtype=np.uint32)
    col_idx = np.empty((n_elements), dtype=np.uint32)

    row_idx[0] = 0
    for i in range(n_samples):
        row_idx[i+1] = row_idx[i]
        for j in range(n_trees):
            if samples_nodes[i,j] == X_nodes_k[j]:
                col_cnt[j] += inbag_samples[i,j]
                col_idx[row_idx[i+1]] = j
                row_idx[i+1] += 1

    col_weight = np.empty((n_trees), dtype=np.float64)
    for j in range(n_trees):
        col_weight[j] = 1. / col_cnt[j]

    weighted_mean = np.empty((n_samples), dtype=np.float64)
    for i in range(n_samples):
        s = 0.
        for jj in range(row_idx[i], row_idx[i+1]):
            s += inbag_samples[i, col_idx[jj]] * col_weight[col_idx[jj]]
        weighted_mean[i] = s / n_trees

    return weighted_mean


@nb.njit("float64[:](int64[:,:], int64[:])", nogil=True, cache=True, parallel=False)
def _compute_weight_with_original_data(samples_nodes, X_nodes_k):
    """
    Function to compute the averaged weight of each individual of the original
    training sample in the forest.
    """
    
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_elements = samples_nodes.size
    
    col_cnt = np.zeros((n_trees), dtype=np.uint32)
    row_idx = np.empty((n_samples + 1), dtype=np.uint32)
    col_idx = np.empty((n_elements), dtype=np.uint32)

    row_idx[0] = 0
    for i in range(n_samples):
        row_idx[i+1] = row_idx[i]
        for j in range(n_trees):
            if samples_nodes[i,j] == X_nodes_k[j]:
                col_cnt[j] += 1
                col_idx[row_idx[i+1]] = j
                row_idx[i+1] += 1

    col_weight = np.empty((n_trees), dtype=np.float64)
    for j in range(n_trees):
        col_weight[j] = 1. / col_cnt[j]

    weighted_mean = np.empty((n_samples), dtype=np.float64)
    for i in range(n_samples):
        s = 0.
        for jj in range(row_idx[i], row_idx[i+1]):
            s += col_weight[col_idx[jj]]
        weighted_mean[i] = s / n_trees

    return weighted_mean


# --------------------------------------
# For the class QuantileRegressionForest
# --------------------------------------

@nb.njit("void(float64[:,:], float64[:], int64[:,:], uint32[:,:], int64[:,:], boolean)", nogil=True, cache=True, parallel=True)
def _compute_conditional_CDF_with_Weighted_approach(C_CDF,
                                                    output_samples,
                                                    samples_nodes,
                                                    inbag_samples,
                                                    X_nodes,
                                                    used_bootstrap_samples):
    """
    "Weighted_CDF" : Function to compute the Conditional Cumulative Distribution Function
    thanks to the weights based on the original dataset or the bootstrap samples according
    to the value of parameter "used_bootstrap_samples".
    """
    
    n_C_CDF = C_CDF.shape[0]
    order_statistic = np.argsort(output_samples)

    # For each observation to compute
    for i in nb.prange(n_C_CDF):
        if used_bootstrap_samples:
            # Compute the Boostrap Data based conditional weights associated to each individual
            weight = _compute_weight_with_bootstrap_data(samples_nodes,
                                                         inbag_samples,
                                                         X_nodes[i, :])
        else:
            # Compute the Original Data based conditional weights associated to each individual
            weight = _compute_weight_with_original_data(samples_nodes, X_nodes[i, :])
        
        # Compute the Conditional Cumulative Distribution Function 
        # ":CDF.shape[1]" : CDF.shape[1] necessary because of a bug of numba combined with parallel=True and the condition if
        C_CDF[i, :C_CDF.shape[1]] = np.cumsum(weight[order_statistic])


@nb.njit("void(float64[:,:], float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:], boolean)", nogil=True, cache=True, parallel=True)
def _compute_conditional_quantile_with_Weighted_CDF(quantiles,
                                                    output_samples,
                                                    samples_nodes,
                                                    inbag_samples,
                                                    X_nodes,
                                                    alpha,
                                                    used_bootstrap_samples):
    """
    "Weighted_CDF" : Function to compute the conditional quantiles thanks to the weights
    based on the original dataset or the bootstrap samples according to the value of
    parameter "used_bootstrap_samples".
    """
    
    n_quantiles = quantiles.shape[0]
    order_statistic = np.argsort(output_samples)

    # For each observation to compute
    for i in nb.prange(n_quantiles):
        if used_bootstrap_samples:
            # Compute the Boostrap Data based conditional weights associated to each individual
            weight = _compute_weight_with_bootstrap_data(samples_nodes,
                                                         inbag_samples,
                                                         X_nodes[i, :])
        else:
            # Compute the Original Data based conditional weights associated to each individual
            weight = _compute_weight_with_original_data(samples_nodes, X_nodes[i, :])
        
        # Compute the quantiles thanks to the Cumulative Distribution Function 
        # for each value of alpha
        CDF = np.cumsum(weight[order_statistic])
   
        # ":quantiles.shape[1]" : quantiles.shape[1] necessary because of a bug of numba combined with parallel=True and the condition if
        quantiles[i, :quantiles.shape[1]] = np.array([
                            output_samples[
                                order_statistic[
                                    np.argmax((CDF >= alpha_var).astype(np.uint32))]
                                          ] 
                            for alpha_var in alpha])


@nb.njit("void(float64[:,:], float64[:], int64[:,:], uint32[:,:], float64[:])", nogil=True, cache=True, parallel=True)
def _compute_conditional_quantile_for_oob_samples_with_Weighted_CDF_using_bootstrap_data(quantiles,
                                                                                         output_samples,
                                                                                         samples_nodes,
                                                                                         inbag_samples,
                                                                                         alpha):
    """
    Predictions and scores OOB by using the weights based on the bootstrap samples 
    for calculating the conditional quantiles.
    """
    
    n_samples = samples_nodes.shape[0]
    order_statistic = np.argsort(output_samples)
    
    # Calculate the conditional quantile of each observation of the original sample with the OOB samples
    for i in nb.prange(n_samples):
        trees_built_without_obs_i_in_bootstrap_samples = np.where(inbag_samples[i] == 0)[0]
        
        if trees_built_without_obs_i_in_bootstrap_samples.size != 0:
            
            # Compute the Boostrap Data based conditional weights associated to each individual
            weight = _compute_weight_with_bootstrap_data(samples_nodes[:, trees_built_without_obs_i_in_bootstrap_samples],
                                                         inbag_samples[:, trees_built_without_obs_i_in_bootstrap_samples],
                                                         samples_nodes[i][trees_built_without_obs_i_in_bootstrap_samples])
            
            # Compute the quantiles thanks to the Cumulative Distribution Function 
            # for each value of alpha
            CDF = np.cumsum(weight[order_statistic])
            
            # ":quantiles.shape[1]" : quantiles.shape[1] necessary because of a bug of numba combined with parallel=True and the condition if
            quantiles[i, :quantiles.shape[1]] = np.array([
                output_samples[
                    order_statistic[
                        np.argmax((CDF >= alpha_var).astype(np.uint32))]
                    ]
                for alpha_var in alpha])


@nb.njit("void(float64[:,:], float64[:], int64[:,:], uint32[:,:], float64[:])", nogil=True, cache=True, parallel=True)
def _compute_conditional_quantile_for_oob_samples_with_Weighted_CDF_using_original_data(quantiles,
                                                                                        output_samples,
                                                                                        samples_nodes,
                                                                                        inbag_samples,
                                                                                        alpha):
    """
    Predictions and scores OOB by using the weights based on the original sample
    for calculating the conditional quantiles.
    """
    
    n_samples = samples_nodes.shape[0]
    idx_n_samples = np.arange(0, n_samples)
    order_statistic = np.argsort(output_samples)
    
    # Calculate the conditional quantile of each observation of the original sample with the OOB samples
    for i in nb.prange(n_samples):
        trees_built_without_obs_i_in_bootstrap_samples = np.where(inbag_samples[i] == 0)[0]
        
        if trees_built_without_obs_i_in_bootstrap_samples.size != 0:
            
            idx_n_samples_without_original_obs_i = np.delete(idx_n_samples, i)
            
            # Compute the Original Data based conditional weights associated to each individual
            weight = _compute_weight_with_original_data(
                samples_nodes[:, trees_built_without_obs_i_in_bootstrap_samples][idx_n_samples_without_original_obs_i],
                samples_nodes[i][trees_built_without_obs_i_in_bootstrap_samples])
            
            # Compute the quantiles thanks to the Cumulative Distribution Function 
            # for each value of alpha
            order_statistic_without_obs_i = order_statistic[order_statistic != i]
            order_statistic_without_obs_i[order_statistic_without_obs_i > i] = order_statistic_without_obs_i[order_statistic_without_obs_i > i] - 1
            CDF = np.cumsum(weight[order_statistic_without_obs_i])
            
            # ":quantiles.shape[1]" : quantiles.shape[1] necessary because of a bug of numba combined with parallel=True and the condition if
            quantiles[i, :quantiles.shape[1]] = np.array([
                output_samples[idx_n_samples_without_original_obs_i][
                    order_statistic_without_obs_i[
                        np.argmax((CDF >= alpha_var).astype(np.uint32))]
                    ]
                for alpha_var in alpha])


@nb.njit("void(float64[:,:,:], float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:], boolean)", nogil=True, cache=True, parallel=True)
def _compute_conditional_quantile_on_each_tree(quantiles, 
                                               output_samples,
                                               samples_nodes,
                                               idx_bootstrap_samples,
                                               X_nodes,
                                               alpha,
                                               used_bootstrap_samples):
    """
    "Averaged_Quantile" : Function to compute the conditional quantiles with the 
    samples falling into the same leaf node as the conditional observation "x". 
    Use the original or the bootstrap samples to do this according to the value 
    of the parameter "used_bootstrap_samples".
    """

    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_quantiles = X_nodes.shape[0]
    
    X_nodes = np.asfortranarray(X_nodes)

    # Unique leaf nodes of each tree
    samples_nodes = np.asfortranarray(samples_nodes)
    idx_leaves_by_tree = [np.unique(samples_nodes[:, i]) for i in range(n_trees)]
    
    # Compute the conditional quantiles on each tree
    for i in nb.prange(n_trees):
        idx_leaves_for_tree_i = idx_leaves_by_tree[i]
        Y_leaves = np.empty((n_samples), dtype=np.float64)
        idx_X_nodes_in_current_leaf = np.empty((n_quantiles), dtype=np.uint32)
        
        # Browse the leaves of the tree.
        for idx_current_leaf in idx_leaves_for_tree_i:            
            
            # Get the conditional observations X=x falling into the current leaf
            n_idx_X_nodes_in_current_leaf = 0
            for j in range(n_quantiles):
                if X_nodes[j, i] == idx_current_leaf:
                    idx_X_nodes_in_current_leaf[n_idx_X_nodes_in_current_leaf] = j
                    n_idx_X_nodes_in_current_leaf += 1
            
            # Get the observations of the bootstrap samples or the sample original 
            # being in the current leaf
            n_Y_leaves = 0
            if used_bootstrap_samples:
                for k in range(n_samples):
                    if samples_nodes[idx_bootstrap_samples[k, i], i] == idx_current_leaf:
                        Y_leaves[n_Y_leaves] = output_samples[idx_bootstrap_samples[k, i]]
                        n_Y_leaves += 1
            else:
                for k in range(n_samples):
                    if samples_nodes[k, i] == idx_current_leaf:
                        Y_leaves[n_Y_leaves] = output_samples[k]
                        n_Y_leaves += 1
            
            # Compute the conditional quantiles given X=x with the observations in this leaf
            # np.int64(n_Y_leaves): np.int64 necessary because of a bug of numba when using parallel=True
            # ":quantiles.shape[2]" : quantiles.shape[2] necessary because of a bug of numba combined with parallel=True
            quantiles_in_leaf = np.percentile(Y_leaves[:np.int64(n_Y_leaves)], alpha*100)            
            for l in range(n_idx_X_nodes_in_current_leaf):
                quantiles[i, idx_X_nodes_in_current_leaf[l], :quantiles.shape[2]] = quantiles_in_leaf


@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], DictType(int64, uint32[:]), float64[:])", nogil=True, cache=True, parallel=True)
def _compute_conditional_quantile_for_oob_samples_on_each_tree_with_bootstrap_data(output_samples,
                                                                                   samples_nodes,
                                                                                   idx_bootstrap_samples,
                                                                                   idx_oob_samples,
                                                                                   alpha):
    """
    Predictions and scores OOB by using the bootstrap samples associated with 
    each tree for computing the conditional quantiles.
    """

    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_alphas = alpha.shape[0]

    # Unique leaf nodes of each tree
    samples_nodes = np.asfortranarray(samples_nodes)
    idx_leaves_by_tree = [np.unique(samples_nodes[:, i]) for i in range(n_trees)]
    
    quantiles = np.zeros((n_trees, n_samples, n_alphas), dtype=np.float64)
    n_quantiles = np.zeros((n_trees, n_samples), dtype=np.uint32)
    
    # Compute the conditional quantiles of the OOB samples of each tree with the bootstrap samples
    for i in nb.prange(n_trees):
        idx_leaves_for_tree_i = idx_leaves_by_tree[i]
        X_nodes_oob_for_tree_i = samples_nodes[:, i][idx_oob_samples[np.int64(i)]]
        n_X_nodes_oob_for_tree_i = X_nodes_oob_for_tree_i.shape[0]
        Y_leaves = np.empty((n_samples), dtype=np.float64)
        idx_X_nodes_oob_in_current_leaf = np.empty((n_X_nodes_oob_for_tree_i), dtype=np.uint32)
        
        # Browse the leaves of the tree.
        for idx_current_leaf in idx_leaves_for_tree_i:
            
            # Get the OOB samples falling into the current leaf
            n_idx_X_nodes_oob_in_current_leaf = 0
            for j in range(n_X_nodes_oob_for_tree_i):
                if X_nodes_oob_for_tree_i[j] == idx_current_leaf:
                    idx_X_nodes_oob_in_current_leaf[n_idx_X_nodes_oob_in_current_leaf] = j
                    n_idx_X_nodes_oob_in_current_leaf += 1
            
            # Get the observations of the bootstrap samples being in the current leaf
            n_Y_leaves= 0
            for k in range(n_samples):
                if samples_nodes[idx_bootstrap_samples[k, i], i] == idx_current_leaf:
                    Y_leaves[n_Y_leaves] = output_samples[idx_bootstrap_samples[k, i]]
                    n_Y_leaves += 1
            
            # Compute the conditional quantiles given X=x with the observations in this leaf
            # np.int64(n_Y_leaves): np.int64 necessary because of a bug of numba when using parallel=True
            quantiles_in_leaf = np.percentile(Y_leaves[:np.int64(n_Y_leaves)], alpha*100)            
            for l in range(n_idx_X_nodes_oob_in_current_leaf):
                quantiles[i, idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]], :quantiles.shape[2]] = quantiles_in_leaf
                n_quantiles[i, idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]]] = 1
    
    n_quantiles = n_quantiles.sum(axis=0)
    n_quantiles[n_quantiles == 0] = 1
    
    return quantiles.sum(axis=0)/n_quantiles.reshape(-1,1)


@nb.njit("float64[:,:](float64[:], int64[:,:], DictType(int64, uint32[:]), float64[:])", nogil=True, cache=True, parallel=True)
def _compute_conditional_quantile_for_oob_samples_on_each_tree_with_original_data(output_samples,
                                                                                  samples_nodes,
                                                                                  idx_oob_samples,
                                                                                  alpha):
    """
    Predictions and scores OOB by using the original sample for computing the
    conditional quantiles.
    """

    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_alphas = alpha.shape[0]

    # Unique leaf nodes of each tree
    samples_nodes = np.asfortranarray(samples_nodes)
    idx_leaves_by_tree = [np.unique(samples_nodes[:, i]) for i in range(n_trees)]
    
    quantiles = np.zeros((n_trees, n_samples, n_alphas), dtype=np.float64)
    n_quantiles = np.zeros((n_trees, n_samples), dtype=np.uint32)
    
    # Compute the conditional quantiles of the OOB samples of each tree with the original sample
    for i in nb.prange(n_trees):
        idx_leaves_for_tree_i = idx_leaves_by_tree[i]
        X_nodes_oob_for_tree_i = samples_nodes[:, i][idx_oob_samples[np.int64(i)]]
        n_X_nodes_oob_for_tree_i = X_nodes_oob_for_tree_i.shape[0]
        idx_Y_in_current_leaf = np.empty((n_samples), dtype=np.uint32)
        idx_X_nodes_oob_in_current_leaf = np.empty((n_X_nodes_oob_for_tree_i), dtype=np.uint32)
        
        # Browse the leaves of the tree.
        for idx_current_leaf in idx_leaves_for_tree_i:
            
            # Get the OOB samples falling into the current leaf
            n_idx_X_nodes_oob_in_current_leaf = 0
            for j in range(n_X_nodes_oob_for_tree_i):
                if X_nodes_oob_for_tree_i[j] == idx_current_leaf:
                    idx_X_nodes_oob_in_current_leaf[n_idx_X_nodes_oob_in_current_leaf] = j
                    n_idx_X_nodes_oob_in_current_leaf += 1
            
            # Get the observations of the original sample being in the current leaf
            n_idx_Y_in_current_leaf = 0
            for k in range(n_samples):
                if samples_nodes[k, i] == idx_current_leaf:
                    idx_Y_in_current_leaf[n_idx_Y_in_current_leaf] = k
                    n_idx_Y_in_current_leaf += 1
            
            # Compute the conditional quantiles given X=x with the observations in this leaf
            for l in range(n_idx_X_nodes_oob_in_current_leaf):
                quantiles_in_leaf = np.percentile(
                    output_samples[
                        idx_Y_in_current_leaf[:np.int64(n_idx_Y_in_current_leaf)][
                            idx_Y_in_current_leaf[:np.int64(n_idx_Y_in_current_leaf)] != 
                            idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]]]
                        ],
                    alpha*100)
                quantiles[i, idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]], :quantiles.shape[2]] = quantiles_in_leaf
                n_quantiles[i, idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]]] = 1

    n_quantiles = n_quantiles.sum(axis=0)
    n_quantiles[n_quantiles == 0] = 1
    
    return quantiles.sum(axis=0)/n_quantiles.reshape(-1,1)


# ---------------------------------------------------------------
# For the class MinimumConditionalExpectedCheckFunctionWithLeaves
# ---------------------------------------------------------------

@nb.njit("float64(float64[:], float64)", nogil=True, cache=True, parallel=False)
def _averaged_check_function_alpha_float_unparallel(u, alpha):
    """
    Definition of the check function also called pinball loss function.
    """
    
    n_samples = u.shape[0]
    res = 0.    
    for i in range(n_samples):
        res += u[i]*(alpha - (u[i] < 0.))
    
    res /= n_samples
    return res


@nb.njit("float64(float64[:], float64)", nogil=True, cache=True, parallel=True)
def _averaged_check_function_alpha_float_parallel(u, alpha):
    """
    Definition of the check function also called pinball loss function.
    """

    n_samples = u.shape[0]
    res = 0.    
    for i in nb.prange(n_samples):
        res += u[i]*(alpha - (u[i] < 0.))
    
    res /= n_samples
    return res


@nb.njit("float64[:](float64[:,:], float64[:])", nogil=True, cache=True, parallel=True)
def _averaged_check_function_alpha_array(u, alpha):
    """
    Definition of the check function also called pinball loss function.
    """

    n_alphas = alpha.shape[0]
    n_samples = u.shape[0]

    check_function = np.empty((n_alphas, n_samples), dtype=np.float64)
    for i in nb.prange(n_samples):
        for j in range(n_alphas):
            check_function[j,i] = u[i,j]*(alpha[j] - (u[i,j] < 0.))

    averaged_check_function = np.empty((n_alphas), dtype=np.float64)
    for i in nb.prange(n_alphas):
        averaged_check_function[i] = check_function[i,:].mean()

    return averaged_check_function


@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], float64[:], boolean)", nogil=True, cache=True, parallel=True)
def _compute_expected_value_check_function(output_samples, samples_nodes, idx_bootstrap_samples, alpha, used_bootstrap_samples):
    """
    Compute the expected optimal value of the averaged check function.
    """
    
    # Number of trees in the forest 
    n_trees = samples_nodes.shape[1]
    
    # Size of the sample
    n_samples = samples_nodes.shape[0]
    
    # Number of conditional parts to compute
    n_alphas = alpha.size  # Number of probabilities
    
    # Unique leaf nodes of each tree
    samples_nodes = np.asfortranarray(samples_nodes)
    idx_leaves_by_tree = [np.unique(samples_nodes[:,i]) for i in range(n_trees)]
    
    expectation_forest_by_alpha = np.empty((n_alphas, 2), dtype=np.float64) # 2 for the classical and weighted mean
    expectation_by_tree = np.empty((n_alphas, n_trees, 2), dtype=np.float64) # 2 for the classical and weighted mean
    for i in nb.prange(n_trees):
    #for i,idx_leaves_for_tree_i in enumerate(idx_leaves_by_tree):
        idx_leaves_for_tree_i = idx_leaves_by_tree[i]
        n_idx_leaves_for_tree_i = idx_leaves_for_tree_i.shape[0]
        expectation_by_node = np.empty((n_alphas, n_idx_leaves_for_tree_i), dtype=np.float64)
        node_weights = np.empty((n_idx_leaves_for_tree_i), dtype=np.float64)
        Y_leaves = np.empty((n_samples), dtype=np.float64)
        
        # Browse the leaves of the tree.
        for j, idx_current_leaf in enumerate(idx_leaves_for_tree_i):
            n_Y_leaves = 0
            
            # Get the observations of the bootstrap samples or the sample original 
            # being in the current leaf
            if used_bootstrap_samples:
                for k in range(n_samples):
                    if samples_nodes[idx_bootstrap_samples[k, i], i] == idx_current_leaf:
                        Y_leaves[n_Y_leaves] = output_samples[idx_bootstrap_samples[k, i]]
                        n_Y_leaves += 1
            else:
                for k in range(n_samples):
                    if samples_nodes[k, i] == idx_current_leaf:
                        Y_leaves[n_Y_leaves] = output_samples[k]
                        n_Y_leaves += 1
            
            # Compute the minimum of the expected conditional check function within the current leaf
            # np.int64(n_Y_leaves): np.int64 necessary because of a bug of numba when using parallel=True
            Y_leaves_node = Y_leaves[:np.int64(n_Y_leaves)]
            argsort_Y_leaves_node = np.argsort(Y_leaves_node)
            node_weights[j] = Y_leaves_node.shape[0]/n_samples
            m=0
            for l, alpha_temp in enumerate(alpha):
                if m < Y_leaves_node.shape[0]:
                    ans = _averaged_check_function_alpha_float_unparallel(
                                    Y_leaves_node - Y_leaves_node[argsort_Y_leaves_node[m]],
                                    alpha_temp)
                else:
                    for o in range(l, n_alphas):
                        expectation_by_node[o,j] = _averaged_check_function_alpha_float_unparallel(
                                                        Y_leaves_node - Y_leaves_node[argsort_Y_leaves_node[-1]],
                                                        alpha[o])
                    break
                
                if l == 0:
                    m=1
                else:
                    m+=1
                while(m < Y_leaves_node.shape[0]):
                    temp = _averaged_check_function_alpha_float_unparallel(
                                Y_leaves_node - Y_leaves_node[argsort_Y_leaves_node[m]],
                                alpha_temp)
                    if(temp <= ans):
                        ans = temp
                        m+=1
                    else:
                        m-=1
                        break
                expectation_by_node[l,j] = ans
                
        # Mean of the minimum over all leaves of the tree
        for n in range(n_alphas):
            expectation_by_tree[n, i, 0] = expectation_by_node[n, :].sum()/n_idx_leaves_for_tree_i
            expectation_by_tree[n, i, 1] = (expectation_by_node[n, :]*node_weights).sum()
    
    # Mean over all trees
    for i in nb.prange(n_alphas):      
        expectation_forest_by_alpha[i, 0] = expectation_by_tree[i, :, 0].sum()/n_trees # First column: classical mean
        expectation_forest_by_alpha[i, 1] = expectation_by_tree[i, :, 1].sum()/n_trees # Second column: weighted mean
    
    return expectation_forest_by_alpha


# ------------------------------------------------------------------
# For the classes MinimumConditionalExpectedCheckFunctionWithWeights 
# ------------------------------------------------------------------

@nb.njit("float64(float64[:], float64[:], float64, float64)", nogil=True, cache=True, parallel=False)
def _averaged_check_function_alpha_float_unparallel_product_weight(output_samples, weight, theta, alpha):

    n_samples = output_samples.shape[0]
    res = 0.    
    for i in range(n_samples):
        res += (output_samples[i] - theta)*(alpha - ((output_samples[i] - theta) < 0.))*weight[i]
    
    return res


@nb.njit("void(float64[:,:], float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:], boolean)", nogil=True, cache=True, parallel=True)
def _compute_conditional_minimum_expectation(minimum_expectation,
                                             output_samples,
                                             samples_nodes,
                                             inbag_samples,
                                             X_nodes,
                                             alpha,
                                             used_bootstrap_samples):
    
    n_minimum_expectation = minimum_expectation.shape[0]
    argsort_output_samples = np.argsort(output_samples)

    # For each observation to compute
    for i in nb.prange(n_minimum_expectation):
        if used_bootstrap_samples:
            # Compute the Boostrap Data based conditional weights associated to each individual
            weight = _compute_weight_with_bootstrap_data(samples_nodes,
                                                         inbag_samples,
                                                         X_nodes[i, :])
        else:
            # Compute the Original Data based conditional weights associated to each individual
            weight = _compute_weight_with_original_data(samples_nodes, X_nodes[i, :])
        
        # Compute the minimum of the expected conditional check function given X=x
        k=0
        for j, alpha_temp in enumerate(alpha):
            if k < output_samples.shape[0]:
                ans = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                     output_samples,
                                                     weight,
                                                     output_samples[argsort_output_samples[k]],
                                                     alpha_temp)
            else:
                for l in range(j, alpha.shape[0]):
                    minimum_expectation[i, l] = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                         output_samples,
                                                         weight,
                                                         output_samples[argsort_output_samples[-1]],
                                                         alpha[l])
                break
                    
            if j == 0: 
                k=1
            else:
                k+=1
            while(k < output_samples.shape[0]):
                temp = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                 output_samples,
                                                 weight,
                                                 output_samples[argsort_output_samples[k]],
                                                 alpha_temp)
                if(temp <= ans):
                    ans = temp
                    k+=1
                else:
                    k-=1
                    break
            minimum_expectation[i, j] = ans