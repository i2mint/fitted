# fitted.model_equivalence

Functions to gauge learner equivalence

### Functions

| [`estimator_kind`](#fitted.model_equivalence.estimator_kind)(estimator)                        | Returns the kind (string) of an Estimator                                                                                             |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `get_learner`(learner)                                                                            |                                                                                                                                       |
| [`get_model_action_and_equivalence_scorer`](#fitted.model_equivalence.get_model_action_and_equivalence_scorer)(...)     | Returns a default (model_action, equivalence_scorer) for a given model_action It will return the model_action as is if not a learner. |
| `get_xy_factory_for_estimator`(estimator)                                                         |                                                                                                                                       |
| [`learner_equivalence`](#fitted.model_equivalence.learner_equivalence)(learner_1, learner_2[, ...]) | Returns an score that measures how much the two learners are equivalent.                                                              |

### fitted.model_equivalence.estimator_kind(estimator)

Returns the kind (string) of an Estimator

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### fitted.model_equivalence.get_model_action_and_equivalence_scorer(model_action)

Returns a default (model_action, equivalence_scorer) for a given model_action
It will return the model_action as is if not a learner.
If a learner, it will try to figure out a default model_action for it.

### fitted.model_equivalence.learner_equivalence(learner_1, learner_2, xy=None, model_action=None, equivalence_scorer=None)

Returns an score that measures how much the two learners are equivalent.
The user can specify what data to use (`xy`) to fit the learners,
what action to take on the fitted model (`model_action`),
and what function to apply to the two results to compute the final score.

But the user doesn’t HAVE to specify all that usually (if the learners are
all proper sklearn estimators) – instead, the function will try to figure
out defaults for any of these if not given.

```pycon
>>> from sklearn.linear_model import LinearRegression as Regressor
>>> from sklearn.linear_model import RidgeClassifier as Classifier
>>> from sklearn.decomposition import PCA as UnsupervisedTransformer
>>>
>>> from fitted.model_equivalence import learner_equivalence
>>>
>>> for learner in (Regressor, Classifier, UnsupervisedTransformer):
...     # assert that a learner is equivalent to itself
...     assert learner_equivalence(learner, learner)
```
