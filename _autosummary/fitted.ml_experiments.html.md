# fitted.ml_experiments

Lightweight experiment runner for ML model evaluation.

This module provides a minimal interface for running train/test experiments
across multiple datasets and models, with results persistence.

### Functions

| [`cleanup_temp_dir`](#fitted.ml_experiments.cleanup_temp_dir)(temp_dir, \*[, verbose])         | Clean up a temporary directory and print status.            |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [`create_temp_projects_dir`](#fitted.ml_experiments.create_temp_projects_dir)([prefix])                | Create a persistent temporary directory for examples/tests. |
| [`load_experiment_results`](#fitted.ml_experiments.load_experiment_results)(project_store)            | Load experiment results from a project store.               |
| `pjoin`(\*paths)                                                                                   |                                                             |
| [`summarize_results`](#fitted.ml_experiments.summarize_results)(results, \*[, metric, sort_by]) | Generate summary statistics sorted by performance.          |
| [`temp_uri_to_project_store`](#fitted.ml_experiments.temp_uri_to_project_store)(name, \*[, ...])        | Create a project store in a temporary directory.            |
| [`uri_to_project_store`](#fitted.ml_experiments.uri_to_project_store)(name, \*[, projects_dir])    | Convert a project name to a MutableMapping store.           |

### Classes

| [`ExperimentConfig`](#fitted.ml_experiments.ExperimentConfig)([n_splits, test_size, ...])       | Configuration for experiment runs.                             |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [`ExperimentRunner`](#fitted.ml_experiments.ExperimentRunner)(\*[, project_store, config, ...]) | Runs ML model evaluation experiments with minimal boilerplate. |

### *class* fitted.ml_experiments.ExperimentConfig(n_splits=5, test_size=0.2, random_state=None, scoring='accuracy', verbose=True, n_jobs=-1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Configuration for experiment runs.

```pycon
>>> config = ExperimentConfig(n_splits=3, scoring='accuracy')
>>> config.n_splits
3
```

### *class* fitted.ml_experiments.ExperimentRunner(, project_store=None, config=None, project_dir='/home/runner/.config/fitted/ml_experiments/projects')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Runs ML model evaluation experiments with minimal boilerplate.

```pycon
>>> runner = ExperimentRunner(project_store='test_experiments')
>>> # Run experiments
>>> results = runner.run(
...     datasets={'iris': (X_iris, y_iris)},
...     models={'rf': RandomForestClassifier()}
... )
```

#### *property* project_store *: [MutableMapping](https://docs.python.org/3/library/typing.html#typing.MutableMapping)*

Access to the underlying project store.

#### run(datasets, models, \*\*config_overrides)

Run experiments for all dataset/model combinations.

* **Parameters:**
  * **datasets** (`Union`[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)], [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)]]]) – Mapping or iterable of (name, (X, y)) pairs
  * **models** (`Union`[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]]) – Mapping or iterable of (name, model) pairs
  * **\*\*config_overrides** – Override config parameters for this run
* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/typing.html#typing.MutableMapping)
* **Returns:**
  The project_store containing all results

### fitted.ml_experiments.cleanup_temp_dir(temp_dir, , verbose=True)

Clean up a temporary directory and print status.

* **Parameters:**
  * **temp_dir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to the temporary directory to remove
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to print cleanup status
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### Example

```pycon
>>> temp_dir = create_temp_projects_dir()
>>> # ... use temp_dir ...
>>> cleanup_temp_dir(temp_dir)
```

### fitted.ml_experiments.create_temp_projects_dir(prefix='ml_experiments_')

Create a persistent temporary directory for examples/tests.

This creates a temporary directory that persists beyond the function call,
making it suitable for tests and examples that need to demonstrate persistence.

* **Parameters:**
  **prefix** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prefix for the temporary directory name
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Path to the created temporary directory

### Example

```pycon
>>> temp_dir = create_temp_projects_dir('my_test_')
>>> print(f"Using temporary directory: {temp_dir}")
```

### fitted.ml_experiments.load_experiment_results(project_store)

Load experiment results from a project store.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/typing.html#typing.MutableMapping)

```pycon
>>> results = load_experiment_results('my_experiments')
>>> list(results)  # See all experiment keys
['iris/rf', 'iris/svm', ...]
```

### fitted.ml_experiments.summarize_results(results, , metric='accuracy', sort_by='mean')

Generate summary statistics sorted by performance.

Yields (experiment_key, stats_dict) tuples sorted by the specified metric.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

```pycon
>>> for key, stats in summarize_results(results, metric='accuracy'):
...     print(f"{key}: {stats['mean']:.3f} ± {stats['std']:.3f}")
```

### fitted.ml_experiments.temp_uri_to_project_store(name, , temp_projects_dir=None)

Create a project store in a temporary directory.

This is a convenience function for tests and examples that combines
temp directory creation with project store setup.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Project name
  * **temp_projects_dir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional temp directory path. If None, creates one.
* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/typing.html#typing.MutableMapping)
* **Returns:**
  MutableMapping project store in temporary location

### fitted.ml_experiments.uri_to_project_store(name, , projects_dir='/home/runner/.config/fitted/ml_experiments/projects')

Convert a project name to a MutableMapping store.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/typing.html#typing.MutableMapping)

```pycon
>>> store = uri_to_project_store('my_project')
>>> isinstance(store, MutableMapping)
True
```
