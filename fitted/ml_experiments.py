"""Lightweight experiment runner for ML model evaluation.

This module provides a minimal interface for running train/test experiments
across multiple datasets and models, with results persistence.
"""

import os
import tempfile
from functools import partial
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, MutableMapping, Iterator
from collections.abc import Mapping as ABCMapping
from pathlib import Path
import time
from datetime import datetime

import dol
import dill
from config2py import get_app_folder
from sklearn.model_selection import cross_validate, train_test_split
import numpy as np

# --------------------------------------------------------------------------------------
# Default configuration
# --------------------------------------------------------------------------------------

ensure_dir = partial(dol.process_path, ensure_dir_exists=True)
pjoin = lambda *paths: os.path.join(*paths)

PROJECTS_DIR_ENVVAR = "FITTED_DFLT_EXP_PROJECTS_DIR"


def get_pkg_data_rootdir() -> str:
    """The app folder of ``fitted``, created on first use rather than at import."""
    return ensure_dir(get_app_folder("fitted", folder_kind="config"))


def get_default_projects_dir(*, create: bool = True) -> str:
    """The folder holding the projects (one sub-folder each) unless told otherwise.

    It is ``$FITTED_DFLT_EXP_PROJECTS_DIR`` if that is set, otherwise a folder in
    the app folder. Both are looked up when called, not at import, so importing
    ``fitted.ml_experiments`` touches no disk and works with an unwritable HOME.
    """
    from_env = os.getenv(PROJECTS_DIR_ENVVAR)
    if from_env:
        path = os.path.abspath(os.path.expanduser(from_env))
    else:
        path = pjoin(get_pkg_data_rootdir(), "ml_experiments", "projects")
    return ensure_dir(path) if create else path


_LAZY_MODULE_ATTRS = {
    "PKG_DATA_ROOTDIR": get_pkg_data_rootdir,
    "DFLT_PROJECTS_DIR": get_default_projects_dir,
}


def __getattr__(name):
    # ``fitted.ml_experiments.DFLT_PROJECTS_DIR`` keeps working, but is computed on
    # first access instead of at import.
    if name in _LAZY_MODULE_ATTRS:
        return _LAZY_MODULE_ATTRS[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


dill_wrap = dol.Pipe(
    dol.wrap_kvs(value_encoder=dill.dumps, value_decoder=dill.loads),
    dol.mk_dirs_if_missing,
    dol.add_ipython_key_completions,
)
DillFiles = dill_wrap(dol.Files)


def uri_to_project_store(
    name: str, *, projects_dir: str | None = None
) -> MutableMapping:
    """Convert a project name to a MutableMapping store.

    ``projects_dir`` defaults to ``get_default_projects_dir()`` (not looked up if
    ``name`` is an absolute path).

    >>> store = uri_to_project_store('my_project')  # doctest: +SKIP
    >>> isinstance(store, MutableMapping)  # doctest: +SKIP
    True
    """
    if projects_dir is None and not os.path.isabs(os.path.expanduser(name)):
        projects_dir = get_default_projects_dir(create=False)
    project_path = dol.process_path(
        os.path.join(projects_dir or "", name), ensure_dir_exists=True
    )
    return DillFiles(project_path)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration for experiment runs.

    >>> config = ExperimentConfig(n_splits=3, scoring='accuracy')
    >>> config.n_splits
    3
    """

    n_splits: int = 5
    test_size: float = 0.2
    random_state: int | None = None
    scoring: str | list[str] = "accuracy"
    verbose: bool = True
    n_jobs: int = -1


# --------------------------------------------------------------------------------------
# Core experiment logic
# --------------------------------------------------------------------------------------


def _normalize_datasets(
    datasets: Mapping[str, tuple] | Iterable[tuple[str, tuple]],
) -> Iterator[tuple[str, tuple]]:
    """Normalize datasets input to iterator of (name, (X, y)) pairs.

    >>> list(_normalize_datasets({'d1': (X1, y1)}))  # doctest: +SKIP
    [('d1', (X1, y1))]
    >>> list(_normalize_datasets([('d1', (X1, y1))]))  # doctest: +SKIP
    [('d1', (X1, y1))]
    """
    if isinstance(datasets, ABCMapping):
        yield from datasets.items()
    else:
        yield from datasets


def _normalize_models(
    models: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> Iterator[tuple[str, Any]]:
    """Normalize models input to iterator of (name, model) pairs.

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> list(_normalize_models({'rf': RandomForestClassifier()}))  # doctest: +SKIP
    [('rf', RandomForestClassifier(...))]
    """
    if isinstance(models, ABCMapping):
        yield from models.items()
    else:
        yield from models


def _run_single_experiment(
    X: Any,
    y: Any,
    model: Any,
    *,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run a single experiment (cross-validation or train/test split).

    Returns a dictionary with experiment results including scores and metadata.
    """
    start_time = time.time()

    # Prepare scoring metrics
    scoring = config.scoring
    if isinstance(scoring, str):
        scoring = [scoring]

    # Run cross-validation
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=config.n_splits,
        scoring=scoring,
        n_jobs=config.n_jobs,
        return_train_score=True,
    )

    # Compute statistics for each metric
    def _compute_stats(scores: np.ndarray) -> dict[str, float]:
        """Compute mean, std, min, max for scores."""
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    results = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": time.time() - start_time,
        "n_samples": len(X) if hasattr(X, "__len__") else None,
        "n_features": X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else None,
    }

    # Add statistics for each scoring metric
    for metric in scoring:
        test_key = f"test_{metric}"
        train_key = f"train_{metric}"

        if test_key in cv_results:
            results[f"{metric}_test"] = _compute_stats(cv_results[test_key])
        if train_key in cv_results:
            results[f"{metric}_train"] = _compute_stats(cv_results[train_key])

    # Store raw CV results for potential further analysis
    results["cv_results_raw"] = {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in cv_results.items()
    }

    return results


# --------------------------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------------------------


class ExperimentRunner:
    """Runs ML model evaluation experiments with minimal boilerplate.

    >>> runner = ExperimentRunner(project_store='test_experiments')  # doctest: +SKIP
    >>> # Run experiments
    >>> results = runner.run(
    ...     datasets={'iris': (X_iris, y_iris)},
    ...     models={'rf': RandomForestClassifier()}
    ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        project_store: MutableMapping | str | None = None,
        config: ExperimentConfig | None = None,
        project_dir: str | None = None,
    ):
        """Initialize experiment runner.

        Args:
            project_store: Storage for results (MutableMapping, project name str,
                or None for temp directory)
            config: Experiment configuration (uses defaults if None)
            project_dir: Folder in which a ``project_store`` given by name is kept
                (defaults to ``get_default_projects_dir()``)
        """
        self.config = config or ExperimentConfig()
        self._project_dir = project_dir
        self._project_store = self._init_project_store(project_store)

        if self.config.verbose:
            store_location = getattr(self._project_store, "rootdir", "in-memory")
            print(
                f"Experiment runner initialized. Results will be stored in: {store_location}"
            )

    def _init_project_store(
        self, project_store: MutableMapping | str | None
    ) -> MutableMapping:
        """Initialize the project store from various input types."""
        if project_store is None:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="ml_experiments_")
            if self.config.verbose:
                print(
                    f"No project store specified. Using temporary directory: {temp_dir}"
                )
            return DillFiles(temp_dir)
        elif isinstance(project_store, str):
            return uri_to_project_store(project_store, projects_dir=self._project_dir)
        else:
            return project_store

    @property
    def project_store(self) -> MutableMapping:
        """Access to the underlying project store."""
        return self._project_store

    def run(
        self,
        datasets: Mapping[str, tuple] | Iterable[tuple[str, tuple]],
        models: Mapping[str, Any] | Iterable[tuple[str, Any]],
        **config_overrides,
    ) -> MutableMapping:
        """Run experiments for all dataset/model combinations.

        Args:
            datasets: Mapping or iterable of (name, (X, y)) pairs
            models: Mapping or iterable of (name, model) pairs
            **config_overrides: Override config parameters for this run

        Returns:
            The project_store containing all results
        """
        # Create run-specific config
        run_config = ExperimentConfig(**{**asdict(self.config), **config_overrides})

        # Normalize inputs
        datasets_iter = list(_normalize_datasets(datasets))
        models_iter = list(_normalize_models(models))

        total_experiments = len(datasets_iter) * len(models_iter)

        if run_config.verbose:
            print(f"\nStarting {total_experiments} experiments:")
            print(f"  Datasets: {len(datasets_iter)}")
            print(f"  Models: {len(models_iter)}")
            print(f"  CV splits: {run_config.n_splits}")
            print()

        # Run all experiments
        experiment_count = 0
        for dataset_name, (X, y) in datasets_iter:
            for model_name, model in models_iter:
                experiment_count += 1

                if run_config.verbose:
                    print(
                        f"[{experiment_count}/{total_experiments}] "
                        f"Running: {dataset_name} × {model_name}..."
                    )

                # Run experiment
                result = _run_single_experiment(X, y, model, config=run_config)

                # Store result with key: "dataset_name/model_name"
                result_key = f"{dataset_name}/{model_name}"
                self._project_store[result_key] = result

                if run_config.verbose:
                    # Print summary of primary metric
                    primary_metric = (
                        run_config.scoring[0]
                        if isinstance(run_config.scoring, list)
                        else run_config.scoring
                    )
                    test_stats = result.get(f"{primary_metric}_test", {})
                    if test_stats:
                        print(
                            f"  → {primary_metric}: "
                            f"{test_stats['mean']:.4f} ± {test_stats['std']:.4f}"
                        )

        if run_config.verbose:
            print(f"\n✓ All experiments complete. Results stored in project store.")
            store_location = getattr(self._project_store, "rootdir", "in-memory")
            print(f"  Location: {store_location}")

        return self._project_store


# --------------------------------------------------------------------------------------
# Convenience utilities
# --------------------------------------------------------------------------------------


def load_experiment_results(
    project_store: MutableMapping | str, *, projects_dir: str | None = None
) -> MutableMapping:
    """Load experiment results from a project store.

    >>> results = load_experiment_results('my_experiments')  # doctest: +SKIP
    >>> list(results)  # See all experiment keys  # doctest: +SKIP
    ['iris/rf', 'iris/svm', ...]
    """
    if isinstance(project_store, str):
        return uri_to_project_store(project_store, projects_dir=projects_dir)
    return project_store


def summarize_results(
    results: MutableMapping,
    *,
    metric: str = "accuracy",
    sort_by: str = "mean",
) -> Iterable[tuple[str, dict]]:
    """Generate summary statistics sorted by performance.

    Yields (experiment_key, stats_dict) tuples sorted by the specified metric.

    >>> for key, stats in summarize_results(results, metric='accuracy'):  # doctest: +SKIP
    ...     print(f"{key}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    """
    summaries = []

    for key in results:
        result = results[key]
        test_stats = result.get(f"{metric}_test", {})
        if test_stats:
            summaries.append((key, test_stats))

    # Sort by specified stat (default: mean)
    summaries.sort(key=lambda x: x[1].get(sort_by, 0), reverse=True)

    yield from summaries


def create_temp_projects_dir(prefix: str = "ml_experiments_") -> str:
    """Create a persistent temporary directory for examples/tests.

    This creates a temporary directory that persists beyond the function call,
    making it suitable for tests and examples that need to demonstrate persistence.

    Args:
        prefix: Prefix for the temporary directory name

    Returns:
        Path to the created temporary directory

    Example:
        >>> temp_dir = create_temp_projects_dir('my_test_')  # doctest: +SKIP
        >>> print(f"Using temporary directory: {temp_dir}")  # doctest: +SKIP
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    print(f"📁 Created temporary directory: {temp_dir}")
    return temp_dir


def temp_uri_to_project_store(
    name: str, *, temp_projects_dir: str = None
) -> MutableMapping:
    """Create a project store in a temporary directory.

    This is a convenience function for tests and examples that combines
    temp directory creation with project store setup.

    Args:
        name: Project name
        temp_projects_dir: Optional temp directory path. If None, creates one.

    Returns:
        MutableMapping project store in temporary location
    """
    if temp_projects_dir is None:
        temp_projects_dir = create_temp_projects_dir()

    project_path = dol.process_path(
        os.path.join(temp_projects_dir, name), ensure_dir_exists=True
    )
    return DillFiles(project_path)


def cleanup_temp_dir(temp_dir: str, *, verbose: bool = True) -> None:
    """Clean up a temporary directory and print status.

    Args:
        temp_dir: Path to the temporary directory to remove
        verbose: Whether to print cleanup status

    Example:
        >>> temp_dir = create_temp_projects_dir()  # doctest: +SKIP
        >>> # ... use temp_dir ...
        >>> cleanup_temp_dir(temp_dir)  # doctest: +SKIP
    """
    if os.path.exists(temp_dir):
        import shutil

        shutil.rmtree(temp_dir)
        if verbose:
            print(f"🗑️  Cleaned up temporary directory: {temp_dir}")
    elif verbose:
        print(f"⚠️  Temporary directory not found (already cleaned?): {temp_dir}")
