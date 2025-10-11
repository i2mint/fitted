"""Usage examples for ml_experiments module."""

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from fitted.ml_experiments import (
    ExperimentRunner,
    ExperimentConfig,
    load_experiment_results,
    summarize_results,
    create_temp_projects_dir,
    cleanup_temp_dir,
    temp_uri_to_project_store,
)


# --------------------------------------------------------------------------------------
# Example 1: Minimal usage with defaults
# --------------------------------------------------------------------------------------


def example_minimal():
    """Simplest possible usage - just data and models."""
    print("=" * 70)
    print("Example 1: Minimal usage")
    print("=" * 70)

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Create runner (will use temp directory)
    runner = ExperimentRunner()

    # Run experiments
    results = runner.run(
        datasets={'iris': (X, y)},
        models={
            'rf': RandomForestClassifier(random_state=42),
            'svm': SVC(random_state=42),
        },
    )

    # Access results
    print("\nResults for RandomForest:")
    print(results['iris/rf']['accuracy_test'])


# --------------------------------------------------------------------------------------
# Example 2: Multiple datasets
# --------------------------------------------------------------------------------------


def example_multiple_datasets():
    """Compare models across multiple datasets."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple datasets")
    print("=" * 70)

    # Load multiple datasets
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()

    datasets = {
        'iris': (iris.data, iris.target),
        'wine': (wine.data, wine.target),
        'cancer': (cancer.data, cancer.target),
    }

    # Define models
    models = {
        'rf': RandomForestClassifier(random_state=42),
        'gb': GradientBoostingClassifier(random_state=42),
        'lr': LogisticRegression(random_state=42, max_iter=1000),
    }

    # Create temporary project store
    temp_dir = create_temp_projects_dir('example_multi_datasets_')

    try:
        # Run experiments with custom config
        runner = ExperimentRunner(
            project_store=temp_uri_to_project_store(
                'multi_dataset_comparison', temp_projects_dir=temp_dir
            ),
            config=ExperimentConfig(n_splits=10, verbose=True),
        )

        results = runner.run(datasets=datasets, models=models)

        # Summarize results
        print("\n" + "-" * 70)
        print("Summary (sorted by accuracy):")
        print("-" * 70)

        for key, stats in summarize_results(results):
            dataset, model = key.split('/')
            print(f"{dataset:15} {model:10} {stats['mean']:.4f} ± {stats['std']:.4f}")

    finally:
        cleanup_temp_dir(temp_dir)


# --------------------------------------------------------------------------------------
# Example 3: Multiple scoring metrics
# --------------------------------------------------------------------------------------


def example_multiple_metrics():
    """Evaluate models on multiple metrics."""
    print("\n" + "=" * 70)
    print("Example 3: Multiple scoring metrics")
    print("=" * 70)

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Create temporary project store
    temp_dir = create_temp_projects_dir('example_multi_metrics_')

    try:
        runner = ExperimentRunner(
            project_store=temp_uri_to_project_store(
                'multi_metric_eval', temp_projects_dir=temp_dir
            ),
            config=ExperimentConfig(
                n_splits=5,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                verbose=True,
            ),
        )

        results = runner.run(
            datasets={'cancer': (X, y)},
            models={
                'rf': RandomForestClassifier(random_state=42),
                'svm': SVC(random_state=42),
                'lr': LogisticRegression(random_state=42, max_iter=1000),
            },
        )

        # Display all metrics for one model
        print("\n" + "-" * 70)
        print("All metrics for RandomForest on cancer dataset:")
        print("-" * 70)

        rf_result = results['cancer/rf']
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            stats = rf_result[f'{metric}_test']
            print(f"{metric:15} {stats['mean']:.4f} ± {stats['std']:.4f}")

    finally:
        cleanup_temp_dir(temp_dir)


# --------------------------------------------------------------------------------------
# Example 4: Using list/iterable inputs
# --------------------------------------------------------------------------------------


def example_iterable_inputs():
    """Use lists instead of dicts for datasets and models."""
    print("\n" + "=" * 70)
    print("Example 4: Iterable inputs (lists of tuples)")
    print("=" * 70)

    iris = load_iris()
    wine = load_wine()

    # Define as list of tuples
    datasets = [
        ('iris', (iris.data, iris.target)),
        ('wine', (wine.data, wine.target)),
    ]

    models = [
        ('rf_10', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('rf_50', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('rf_100', RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    runner = ExperimentRunner(config=ExperimentConfig(n_splits=3, verbose=True))

    results = runner.run(datasets=datasets, models=models)

    # Compare RF with different n_estimators on iris
    print("\n" + "-" * 70)
    print("Comparing n_estimators for RandomForest on iris:")
    print("-" * 70)

    for n_est in [10, 50, 100]:
        key = f'iris/rf_{n_est}'
        stats = results[key]['accuracy_test']
        print(f"n_estimators={n_est:3d}: {stats['mean']:.4f} ± {stats['std']:.4f}")


# --------------------------------------------------------------------------------------
# Example 5: Config overrides
# --------------------------------------------------------------------------------------


def example_config_overrides():
    """Override config parameters for specific runs."""
    print("\n" + "=" * 70)
    print("Example 5: Config overrides")
    print("=" * 70)

    iris = load_iris()
    X, y = iris.data, iris.target

    # Create temporary project store
    temp_dir = create_temp_projects_dir('example_config_overrides_')

    try:
        # Create runner with default config
        runner = ExperimentRunner(
            project_store=temp_uri_to_project_store(
                'config_override_demo', temp_projects_dir=temp_dir
            ),
            config=ExperimentConfig(n_splits=5, verbose=False),
        )

        models = {
            'rf': RandomForestClassifier(random_state=42),
            'svm': SVC(random_state=42),
        }

        # First run: use default 5 splits, verbose=False
        print("\nRun 1: Using default config (5 splits, verbose=False)")
        results1 = runner.run(datasets={'iris': (X, y)}, models=models)

        # Second run: override to 10 splits and verbose=True
        print("\nRun 2: Overriding to 10 splits and verbose=True")
        results2 = runner.run(
            datasets={'iris': (X, y)}, models=models, n_splits=10, verbose=True
        )

    finally:
        cleanup_temp_dir(temp_dir)


# --------------------------------------------------------------------------------------
# Example 6: Persistence and reloading
# --------------------------------------------------------------------------------------


def example_persistence():
    """Demonstrate saving and reloading results."""
    print("\n" + "=" * 70)
    print("Example 6: Persistence and reloading")
    print("=" * 70)

    iris = load_iris()
    X, y = iris.data, iris.target

    # Create temporary project store
    temp_dir = create_temp_projects_dir('example_persistence_')

    try:
        # Run experiments and save
        print(f"\nRunning experiments and saving to temporary location...")
        runner = ExperimentRunner(
            project_store=temp_uri_to_project_store(
                'my_experiments', temp_projects_dir=temp_dir
            ),
            config=ExperimentConfig(n_splits=5, verbose=False),
        )

        runner.run(
            datasets={'iris': (X, y)},
            models={
                'rf': RandomForestClassifier(random_state=42),
                'svm': SVC(random_state=42),
            },
        )

        # Later: reload results
        print("\nReloading results from the same temporary location...")
        results = temp_uri_to_project_store(
            'my_experiments', temp_projects_dir=temp_dir
        )

        print(f"\nFound {len(list(results))} experiments:")
        for key in results:
            print(f"  - {key}")

        # Analyze results
        print("\nPerformance summary:")
        for key, stats in summarize_results(results):
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    finally:
        cleanup_temp_dir(temp_dir)


# --------------------------------------------------------------------------------------
# Example 7: AI-friendly programmatic generation
# --------------------------------------------------------------------------------------


def example_ai_friendly():
    """Example of how AI might generate experiments programmatically."""
    print("\n" + "=" * 70)
    print("Example 7: AI-friendly programmatic generation")
    print("=" * 70)

    # AI analyzes data and suggests models
    iris = load_iris()
    wine = load_wine()

    # AI determines this is a classification problem with moderate size
    suggested_models = [
        'RandomForestClassifier',
        'GradientBoostingClassifier',
        'SVC',
        'LogisticRegression',
        'DecisionTreeClassifier',
        'GaussianNB',
    ]

    # AI constructs model instances programmatically
    models = {}
    for model_name in suggested_models:
        if model_name == 'RandomForestClassifier':
            models['rf'] = RandomForestClassifier(random_state=42)
        elif model_name == 'GradientBoostingClassifier':
            models['gb'] = GradientBoostingClassifier(random_state=42)
        elif model_name == 'SVC':
            models['svm'] = SVC(random_state=42)
        elif model_name == 'LogisticRegression':
            models['lr'] = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'DecisionTreeClassifier':
            models['dt'] = DecisionTreeClassifier(random_state=42)
        elif model_name == 'GaussianNB':
            models['nb'] = GaussianNB()

    # AI creates datasets dict
    datasets = {
        'iris': (iris.data, iris.target),
        'wine': (wine.data, wine.target),
    }

    # Create temporary project store
    temp_dir = create_temp_projects_dir('example_ai_friendly_')

    try:
        # AI runs experiments
        runner = ExperimentRunner(
            project_store=temp_uri_to_project_store(
                'ai_generated_experiments', temp_projects_dir=temp_dir
            ),
            config=ExperimentConfig(
                n_splits=5, scoring=['accuracy', 'f1_macro'], verbose=True
            ),
        )

        results = runner.run(datasets=datasets, models=models)

        # AI analyzes results and generates report
        print("\n" + "=" * 70)
        print("AI-Generated Analysis Report")
        print("=" * 70)

        for dataset_name in datasets.keys():
            print(f"\n{dataset_name.upper()} Dataset:")
            print("-" * 70)

            # Get results for this dataset
            dataset_results = {
                k.split('/')[1]: results[k]
                for k in results
                if k.startswith(f"{dataset_name}/")
            }

            # Find best model
            best_model = None
            best_accuracy = 0

            for model_name, result in dataset_results.items():
                accuracy = result['accuracy_test']['mean']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name

            print(f"Best model: {best_model} (accuracy: {best_accuracy:.4f})")
            print("\nAll models ranked by accuracy:")

            ranked = sorted(
                dataset_results.items(),
                key=lambda x: x[1]['accuracy_test']['mean'],
                reverse=True,
            )

            for i, (model_name, result) in enumerate(ranked, 1):
                acc_stats = result['accuracy_test']
                f1_stats = result['f1_macro_test']
                print(
                    f"  {i}. {model_name:5} - "
                    f"Acc: {acc_stats['mean']:.4f}±{acc_stats['std']:.4f}, "
                    f"F1: {f1_stats['mean']:.4f}±{f1_stats['std']:.4f}"
                )

    finally:
        cleanup_temp_dir(temp_dir)


# --------------------------------------------------------------------------------------
# Example 8: Custom MutableMapping store
# --------------------------------------------------------------------------------------


def example_custom_store():
    """Use a custom MutableMapping as the project store."""
    print("\n" + "=" * 70)
    print("Example 8: Custom in-memory store")
    print("=" * 70)

    # Simple in-memory store for demonstration
    class InMemoryStore(dict):
        """Simple dict-based store with logging."""

        def __setitem__(self, key, value):
            print(f"  Storing: {key}")
            super().__setitem__(key, value)

    iris = load_iris()
    X, y = iris.data, iris.target

    # Use custom store
    custom_store = InMemoryStore()

    runner = ExperimentRunner(
        project_store=custom_store, config=ExperimentConfig(n_splits=3, verbose=False)
    )

    print("\nRunning experiments with custom store:")
    results = runner.run(
        datasets={'iris': (X, y)},
        models={
            'rf': RandomForestClassifier(random_state=42),
            'svm': SVC(random_state=42),
        },
    )

    print(f"\nStore now contains {len(results)} experiments")
    print(f"Store is the same object: {results is custom_store}")


# --------------------------------------------------------------------------------------
# Run all examples
# --------------------------------------------------------------------------------------


def run_all_examples():
    """Run all examples sequentially."""
    print("=" * 80)
    print("🚀 STARTING ML EXPERIMENTS EXAMPLES")
    print("=" * 80)
    print(
        "📝 All examples use temporary directories and will be cleaned up automatically"
    )
    print()

    example_minimal()
    example_multiple_datasets()
    example_multiple_metrics()
    example_iterable_inputs()
    example_config_overrides()
    example_persistence()
    example_ai_friendly()
    example_custom_store()

    print("\n" + "=" * 80)
    print("🎉 ALL EXAMPLES COMPLETED!")
    print("=" * 80)
    print("✨ All temporary directories have been cleaned up")
    print()


if __name__ == '__main__':
    # Run a specific example
    # example_minimal()

    # Or run all examples
    run_all_examples()
