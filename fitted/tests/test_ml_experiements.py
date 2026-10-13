"""Tests for ml_experiments module."""

import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from fitted.ml_experiments import (
    ExperimentRunner,
    ExperimentConfig,
    load_experiment_results,
    summarize_results,
    uri_to_project_store,
    create_temp_projects_dir,
    cleanup_temp_dir,
)


def test_basic_experiment_run():
    """Test basic experiment execution with synthetic data."""
    print("\n" + "=" * 60)
    print("🧪 Running test_basic_experiment_run")
    print("=" * 60)

    # Create synthetic dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # Create temporary store
    temp_dir = create_temp_projects_dir('test_basic_')

    try:
        # Initialize runner
        runner = ExperimentRunner(
            project_store=temp_dir, config=ExperimentConfig(n_splits=3, verbose=False)
        )

        # Run experiments
        results = runner.run(
            datasets={'synthetic': (X, y)},
            models={
                'rf': RandomForestClassifier(random_state=42),
                'svm': SVC(random_state=42),
            },
        )

        # Verify results
        assert 'synthetic/rf' in results
        assert 'synthetic/svm' in results

        # Check result structure
        rf_result = results['synthetic/rf']
        assert 'accuracy_test' in rf_result
        assert 'mean' in rf_result['accuracy_test']
        assert 'std' in rf_result['accuracy_test']
        assert 'timestamp' in rf_result
        assert 'duration_seconds' in rf_result

        print("✅ Basic experiment run test passed")

    finally:
        cleanup_temp_dir(temp_dir)


def test_multiple_datasets():
    """Test experiments across multiple datasets."""
    print("\n" + "=" * 60)
    print("🧪 Running test_multiple_datasets")
    print("=" * 60)

    # Create multiple datasets
    X1, y1 = make_classification(n_samples=100, n_features=10, random_state=1)
    X2, y2 = make_classification(n_samples=150, n_features=15, random_state=2)

    temp_dir = create_temp_projects_dir('test_multi_datasets_')

    try:
        runner = ExperimentRunner(
            project_store=temp_dir, config=ExperimentConfig(n_splits=2, verbose=False)
        )

        results = runner.run(
            datasets={
                'dataset1': (X1, y1),
                'dataset2': (X2, y2),
            },
            models={'rf': RandomForestClassifier(random_state=42)},
        )

        # Verify all combinations exist
        assert 'dataset1/rf' in results
        assert 'dataset2/rf' in results

        print("✅ Multiple datasets test passed")

    finally:
        cleanup_temp_dir(temp_dir)


def test_multiple_scoring_metrics():
    """Test with multiple scoring metrics."""
    print("\n" + "=" * 60)
    print("🧪 Running test_multiple_scoring_metrics")
    print("=" * 60)

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    temp_dir = create_temp_projects_dir('test_multi_metrics_')

    try:
        runner = ExperimentRunner(
            project_store=temp_dir,
            config=ExperimentConfig(
                n_splits=3,
                scoring=['accuracy', 'f1', 'precision', 'recall'],
                verbose=False,
            ),
        )

        results = runner.run(
            datasets={'data': (X, y)},
            models={'lr': LogisticRegression(random_state=42)},
        )

        result = results['data/lr']

        # Verify all metrics are present
        assert 'accuracy_test' in result
        assert 'f1_test' in result
        assert 'precision_test' in result
        assert 'recall_test' in result

        print("✅ Multiple scoring metrics test passed")

    finally:
        cleanup_temp_dir(temp_dir)


def test_config_overrides():
    """Test that config overrides work correctly."""
    print("\n" + "=" * 60)
    print("🧪 Running test_config_overrides")
    print("=" * 60)

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    temp_dir = create_temp_projects_dir('test_config_overrides_')

    try:
        # Default config with 5 splits
        runner = ExperimentRunner(
            project_store=temp_dir, config=ExperimentConfig(n_splits=5, verbose=False)
        )

        # Override to 2 splits for this run
        results = runner.run(
            datasets={'data': (X, y)},
            models={'rf': RandomForestClassifier(random_state=42)},
            n_splits=2,
        )

        result = results['data/rf']

        # Check that 2 splits were used (2 scores in raw results)
        assert len(result['cv_results_raw']['test_accuracy']) == 2

        print("✅ Config overrides test passed")

    finally:
        cleanup_temp_dir(temp_dir)


def test_iterable_inputs():
    """Test that iterable inputs work (not just mappings)."""
    print("\n" + "=" * 60)
    print("🧪 Running test_iterable_inputs")
    print("=" * 60)

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    temp_dir = create_temp_projects_dir('test_iterable_inputs_')

    try:
        runner = ExperimentRunner(
            project_store=temp_dir, config=ExperimentConfig(n_splits=2, verbose=False)
        )

        # Use lists of tuples instead of dicts
        results = runner.run(
            datasets=[('data1', (X, y))],
            models=[('model1', RandomForestClassifier(random_state=42))],
        )

        assert 'data1/model1' in results

        print("✅ Iterable inputs test passed")

    finally:
        cleanup_temp_dir(temp_dir)


def test_string_project_store():
    """Test using a string as project_store name."""
    print("\n" + "=" * 60)
    print("🧪 Running test_string_project_store")
    print("=" * 60)

    X, y = make_classification(n_samples=50, n_features=10, random_state=42)

    # Create temp directory and use unique project name within it
    temp_base_dir = create_temp_projects_dir('test_string_store_base_')
    project_name = f'test_project_{np.random.randint(10000)}'

    try:
        # Use temp_uri_to_project_store to create project in temp location
        from fitted.ml_experiments import temp_uri_to_project_store

        runner = ExperimentRunner(
            project_store=temp_uri_to_project_store(
                project_name, temp_projects_dir=temp_base_dir
            ),
            config=ExperimentConfig(n_splits=2, verbose=False),
        )

        results = runner.run(
            datasets={'data': (X, y)},
            models={'rf': RandomForestClassifier(random_state=42)},
        )

        assert 'data/rf' in results

        # Load results using the same temp location
        loaded_results = temp_uri_to_project_store(
            project_name, temp_projects_dir=temp_base_dir
        )
        assert 'data/rf' in loaded_results

        print("✅ String project store test passed")

    finally:
        cleanup_temp_dir(temp_base_dir)


def test_summarize_results():
    """Test the summarize_results utility function."""
    print("\n" + "=" * 60)
    print("🧪 Running test_summarize_results")
    print("=" * 60)

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    temp_dir = create_temp_projects_dir('test_summarize_results_')

    try:
        runner = ExperimentRunner(
            project_store=temp_dir, config=ExperimentConfig(n_splits=3, verbose=False)
        )

        results = runner.run(
            datasets={'data': (X, y)},
            models={
                'rf': RandomForestClassifier(random_state=42),
                'svm': SVC(random_state=42),
                'lr': LogisticRegression(random_state=42),
            },
        )

        # Get summary sorted by mean accuracy
        summary = list(summarize_results(results, metric='accuracy', sort_by='mean'))

        # Should have 3 results
        assert len(summary) == 3

        # Each summary should have key and stats
        for key, stats in summary:
            assert isinstance(key, str)
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats

        # Verify sorting (means should be in descending order)
        means = [stats['mean'] for _, stats in summary]
        assert means == sorted(means, reverse=True)

        print("✅ Summarize results test passed")

    finally:
        cleanup_temp_dir(temp_dir)


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🚀 STARTING ML EXPERIMENTS TESTS")
    print("=" * 80)
    print("📝 All temporary directories will be created and cleaned up automatically")
    print()

    test_basic_experiment_run()
    test_multiple_datasets()
    test_multiple_scoring_metrics()
    test_config_overrides()
    test_iterable_inputs()
    test_string_project_store()
    test_summarize_results()

    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 80)
    print("✨ All temporary directories have been cleaned up")
    print()


if __name__ == '__main__':
    run_all_tests()


def test_import_does_not_create_folders_in_the_cwd(tmp_path):
    """The default projects dir is absolute (fitted#2)."""
    import subprocess
    import sys

    code = "import fitted.ml_experiments as m, os; print(os.path.isabs(m.DFLT_PROJECTS_DIR))"
    out = subprocess.run(
        [sys.executable, '-c', code], cwd=tmp_path, capture_output=True, text=True,
        env={k: v for k, v in os.environ.items() if k != 'FITTED_DFLT_EXP_PROJECTS_DIR'},
    )
    assert out.stdout.strip() == 'True', out.stderr
    assert list(tmp_path.iterdir()) == []
