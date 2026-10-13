
# fitted

Machine learning tools

To install:	```pip install fitted```

[Documentation here](https://otosense.github.io/fitted/)


# ml_experiments

Lightweight experiment runner for ML model evaluation with minimal boilerplate.

## Features

- **Minimal boilerplate**: Just provide data and models, get results
- **Flexible storage**: Pluggable `MutableMapping` backends for result persistence
- **Multiple datasets/models**: Automatically runs all combinations
- **Comprehensive metrics**: Built-in cross-validation with multiple scoring metrics
- **AI-friendly**: Simple, predictable API for programmatic use
- **Functional design**: Generator-based, dependency injection, SSOT principles

## Quick Start

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from fitted.ml_experiments import ExperimentRunner

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Run experiments
runner = ExperimentRunner()
results = runner.run(
    datasets={'iris': (X, y)},
    models={
        'rf': RandomForestClassifier(random_state=42),
        'svm': SVC(random_state=42),
    }
)

# Access results
print(results['iris/rf']['accuracy_test'])
# {'mean': 0.96, 'std': 0.02, 'min': 0.93, 'max': 0.98}
```

## Core Concepts

### ExperimentRunner

The main entry point for running experiments. Initialize once, run many times.

```python
runner = ExperimentRunner(
    project_store='my_experiments',  # str, MutableMapping, or None
    config=ExperimentConfig(n_splits=5, verbose=True)
)
```

**Project Store Options:**
- `None`: Uses temporary directory (printed at start)
- `str`: Project name → creates persistent store in your default projects directory
- `MutableMapping`: Any custom key-value store

To see what your default `projects` directory is, see 

```py
from fitted.ml_experiments import DFLT_PROJECTS_DIR
print(DFLT_PROJECTS_DIR)
```

It lives in fitted's app folder (not in the current directory). You can change this default by specifying a different one in the `FITTED_DFLT_EXP_PROJECTS_DIR` environment variable (use an absolute path; a relative one is resolved against the current directory).


### ExperimentConfig

Configuration for experiment runs:

```python
from fitted.ml_experiments import ExperimentConfig

config = ExperimentConfig(
    n_splits=5,              # Cross-validation splits
    test_size=0.2,           # Train/test split ratio
    random_state=42,         # For reproducibility
    scoring='accuracy',      # Single metric or list
    verbose=True,            # Print progress
    n_jobs=-1               # Parallel CV folds
)
```

### Result Structure

Each experiment result is a dict with:

```python
{
    'config': {...},                    # Experiment configuration
    'timestamp': '2025-10-10T14:30:00', # When run
    'duration_seconds': 1.23,           # Runtime
    'n_samples': 150,                   # Dataset size
    'n_features': 4,                    # Feature count
    'accuracy_test': {                  # Stats for each metric
        'mean': 0.96,
        'std': 0.02,
        'min': 0.93,
        'max': 0.98
    },
    'accuracy_train': {...},            # Training scores
    'cv_results_raw': {...}             # Raw sklearn output
}
```

## Usage Patterns

### Multiple Datasets

```python
datasets = {
    'iris': (X_iris, y_iris),
    'wine': (X_wine, y_wine),
    'cancer': (X_cancer, y_cancer),
}

results = runner.run(datasets=datasets, models=models)
```

### Multiple Metrics

```python
runner = ExperimentRunner(
    config=ExperimentConfig(
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
)
```

### Config Overrides

```python
# Default: 5 splits
runner = ExperimentRunner(config=ExperimentConfig(n_splits=5))

# Override for specific run
results = runner.run(datasets, models, n_splits=10, verbose=False)
```

### Iterable Inputs (Alternative to Dicts)

```python
datasets = [
    ('dataset1', (X1, y1)),
    ('dataset2', (X2, y2)),
]

models = [
    ('model1', RandomForestClassifier()),
    ('model2', SVC()),
]

results = runner.run(datasets=datasets, models=models)
```

### Persistence and Reloading

```python
# First session: run and save
runner = ExperimentRunner(project_store='my_project')
runner.run(datasets, models)

# Later session: reload results
from fitted.ml_experiments import load_experiment_results

results = load_experiment_results('my_project')
for key in results:
    print(f"{key}: {results[key]['accuracy_test']['mean']:.3f}")
```

### Result Analysis

```python
from fitted.ml_experiments import summarize_results

# Get sorted summary
for key, stats in summarize_results(results, metric='accuracy'):
    dataset, model = key.split('/')
    print(f"{dataset} × {model}: {stats['mean']:.4f} ± {stats['std']:.4f}")
```

## Custom Storage Backends

You can provide any `MutableMapping` as the project store:

```python
class S3Store(MutableMapping):
    """Store results in S3."""
    def __init__(self, bucket_name):
        self.bucket = boto3.resource('s3').Bucket(bucket_name)
    
    def __setitem__(self, key, value):
        self.bucket.put_object(
            Key=key,
            Body=dill.dumps(value)
        )
    
    def __getitem__(self, key):
        obj = self.bucket.Object(key).get()
        return dill.loads(obj['Body'].read())
    
    # ... implement __delitem__, __iter__, __len__

# Use it
runner = ExperimentRunner(project_store=S3Store('my-bucket'))
```

## AI Assistant Usage

This tool is designed to be easily used by AI assistants. Example prompt:

```
I have this dataset [description]. Please:
1. Suggest appropriate models
2. Run experiments using ml_experiments
3. Analyze results and give recommendations
```

The AI can programmatically:
- Determine appropriate models based on data characteristics
- Construct the models dict
- Run experiments
- Parse and analyze results
- Generate reports

## Architecture

The library follows these principles:

- **Functional over OOP**: Prefer generators and pure functions
- **SOLID principles**: Single responsibility, dependency injection
- **Mapping interface**: Results are `MutableMapping` instances
- **Generator-based**: Lazy evaluation where appropriate
- **Minimal dependencies**: Only sklearn, dol, dill, numpy

## API Reference

### Core Classes

**`ExperimentRunner`**
- `__init__(*, project_store, config)`
- `run(datasets, models, **config_overrides) -> MutableMapping`
- `project_store` property

**`ExperimentConfig`** (dataclass)
- `n_splits: int = 5`
- `test_size: float = 0.2`
- `random_state: int | None = None`
- `scoring: str | list[str] = 'accuracy'`
- `verbose: bool = True`
- `n_jobs: int = -1`

### Utility Functions

**`load_experiment_results(project_store)`**
- Load results from a saved project

**`summarize_results(results, *, metric, sort_by)`**
- Generate sorted summary statistics

**`uri_to_project_store(name, *, projects_dir)`**
- Convert project name to MutableMapping store

## Examples

See `examples.py` for comprehensive usage examples including:
- Minimal usage
- Multiple datasets
- Multiple metrics
- Config overrides
- Persistence
- AI-friendly programmatic generation
- Custom storage backends

## Testing

```bash
python test_ml_experiments.py
```




