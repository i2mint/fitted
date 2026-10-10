"""Internal utils"""
from typing import Union, Tuple
from collections.abc import Iterable, Callable
from sklearn.base import BaseEstimator, TransformerMixin

Learner = BaseEstimator  # but not necessarily fitted
Model = BaseEstimator  # but fitted
Estimator = Union[Callable, str, type, BaseEstimator]
XY = tuple[Iterable, Iterable]
XYFactory = Callable[[], XY]
