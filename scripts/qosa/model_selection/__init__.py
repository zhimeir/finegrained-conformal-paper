# -*- coding: utf-8 -*-

"""
    Add Dosctring of the package
"""


from .validation import cross_validation_forest, cross_validation_kernel
from ._validation import cross_validate

__all__ = ['cross_validation_forest', 'cross_validation_kernel']
