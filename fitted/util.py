"""Internal utils"""
from typing import Iterable, Callable, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin

Learner = BaseEstimator  # but not necessarily fitted
Model = BaseEstimator  # but fitted
Estimator = Union[Callable, str, type, BaseEstimator]
XY = Tuple[Iterable, Iterable]
XYFactory = Callable[[], XY]
