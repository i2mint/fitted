"""
Functions to gauge learner equivalence
"""

from typing import Iterable, Callable

import numpy as np
from sklearn.base import is_regressor, is_classifier, is_outlier_detector
from sklearn.datasets import make_classification, make_regression
from sklearn.base import BaseEstimator, TransformerMixin

from fitted.util import (
    Learner,
    Model,
    Estimator,
    XY,
    XYFactory,
)


def learner_equivalence(
    learner_1, learner_2, xy=None, model_action=None, equivalence_scorer=None
):
    """Returns an score that measures how much the two learners are equivalent.
    The user can specify what data to use (`xy`) to fit the learners,
    what action to take on the fitted model (`model_action`),
    and what function to apply to the two results to compute the final score.

    But the user doesn't HAVE to specify all that usually (if the learners are
    all proper sklearn estimators) -- instead, the function will try to figure
    out defaults for any of these if not given.

    >>> from sklearn.linear_model import LinearRegression as Regressor
    >>> from sklearn.linear_model import RidgeClassifier as Classifier
    >>> from sklearn.decomposition import PCA as UnsupervisedTransformer
    >>>
    >>> from fitted.model_equivalence import learner_equivalence
    >>>
    >>> for learner in (Regressor, Classifier, UnsupervisedTransformer):
    ...     # assert that a learner is equivalent to itself
    ...     assert learner_equivalence(learner, learner)
    """
    # preprocess inputs
    learner_1 = get_learner(learner_1)
    learner_2 = get_learner(learner_2)
    if xy is None:
        xy = get_xy_factory_for_estimator(learner_1)
    if isinstance(xy, Callable):
        X, y = xy()
    else:
        X, y = xy
    model_action = model_action or learner_1
    model_action, dflt_equivalence_scorer = get_model_action_and_equivalence_scorer(
        model_action
    )
    equivalence_scorer = equivalence_scorer or dflt_equivalence_scorer
    # do the stuff
    learner_1.fit(X, y)
    learner_2.fit(X, y)
    learner_1_output = model_action(learner_1, X)
    learner_2_output = model_action(learner_2, X)
    return equivalence_scorer(learner_1_output, learner_2_output)


def _is_estimator_factory(estimator) -> bool:
    """Tells us if the input might be able to be called to get an estimator"""
    return isinstance(estimator, type) or (
        isinstance(estimator, Callable) and not isinstance(estimator, BaseEstimator)
    )


valid_estimator_kinds = {'classifier', 'regressor', 'transformer'}


def estimator_kind(estimator: Estimator) -> str:
    """Returns the kind (string) of an Estimator"""
    if _is_estimator_factory(estimator):
        estimator = estimator()
    if isinstance(estimator, BaseEstimator):
        if is_classifier(estimator):
            kind = 'classifier'
        elif is_regressor(estimator):
            kind = 'regressor'
        elif TransformerMixin in type(estimator).mro():
            kind = 'transformer'
        else:
            kind = 'regressor'  # we'll just use that for unsupervised?
    elif isinstance(estimator, str):
        kind = estimator
    else:
        raise ValueError(f"Couldn't result estimator to a kind: {estimator}")
    return kind


def get_xy_factory_for_estimator(estimator: Estimator) -> XYFactory:
    kind = estimator_kind(estimator)

    if isinstance(kind, str):
        data_generators = {
            'classifier': make_classification,
            'regressor': make_regression,
            'transformer': make_regression,  # or classification better?
        }
        data_gen = data_generators.get(kind, None)
        if data_gen is None:
            raise ValueError(f'A string estimator must be one of {data_generators}')
    elif isinstance(kind, Callable):
        data_gen = kind
    else:
        raise ValueError(f'Unrecognized kind of estimator: {estimator}')
    return data_gen


def get_learner(learner) -> Learner:
    if isinstance(learner, type):
        # TODO: Do this for any Callable? (Need distinguish with callable instance)
        learner = learner()
    return learner


def get_model_action_and_equivalence_scorer(model_action):
    """Returns a default (model_action, equivalence_scorer) for a given model_action
    It will return the model_action as is if not a learner.
    If a learner, it will try to figure out a default model_action for it.
    """
    if isinstance(model_action, Learner):
        learner = model_action
        _estimator_kind = estimator_kind(learner)
        return {
            'classifier': (lambda model, X: getattr(model, 'predict')(X), np.allclose,),
            'regressor': (lambda model, X: getattr(model, 'predict')(X), np.allclose),
            'transformer': (
                lambda model, X: getattr(model, 'transform')(X),
                np.allclose,
            ),
        }.get(_estimator_kind)
    else:
        return model_action, np.allclose
