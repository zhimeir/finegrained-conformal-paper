# -*- coding: utf-8 -*-

"""
Add Dosctring of the package
"""


__version__ = '0.0.1'


from .base_forest import (MinimumConditionalExpectedCheckFunctionWithLeaves, 
                          MinimumConditionalExpectedCheckFunctionWithWeights,
                          QuantileRegressionForest)
from .base_kernel import (MinimumConditionalExpectedCheckFunctionKernel,
                          UnivariateQuantileRegressionKernel)
from .methods_for_qosa import *
from .model import ProbabilisticModel
from .model_selection import (cross_validate, cross_validation_forest, 
                              cross_validation_kernel)
from .qosa import MinimumBasedQosaIndices, QuantileBasedQosaIndices
from .qose import QoseIndices


__all__ = ['MinimumBasedQosaIndices', 'ProbabilisticModel', 'QuantileBasedQosaIndices',
           'QuantileRegressionForest', 'qosa_Quantile__Averaged_Quantile', 
           'qosa_Quantile__Kernel_CDF', 'qosa_Quantile__Weighted_CDF',
           'qosa_Min__Kernel_Min', 'qosa_Min__Min_in_Leaves', 
           'qosa_Min__Weighted_Min', 'qosa_Min__Weighted_Min_with_complete_forest',
           'QoseIndices']