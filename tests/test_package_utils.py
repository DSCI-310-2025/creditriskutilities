import os
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from creditriskutilities import (
    apply_mappings,
    create_output_dir,
    evaluate_model,
    plot_feature_importance,
    compare_models,
)

# Fixtures

@pytest.fixture
def temp_output_dir(tmpdir):
    """Temporary output directory for file-based function testing."""
    return str(tmpdir.mkdir("test_output"))

@pytest.fixture
def test_data():
    """Synthetic binary classification dataset with 5 features."""
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    feature_names = [f'feature{i}' for i in range(5)]
    return X_train, X_test, y_train, y_test, feature_names

@pytest.fixture
def test_metrics():
    """Create test metrics for model comparison."""
    return [
        {'model_name': 'Model1', 'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6, 'f1': 0.65},
        {'model_name': 'Model2', 'accuracy': 0.9, 'precision': 0.85, 'recall': 0.8, 'f1': 0.82}
    ]

# Tests for apply_mappings

def test_apply_value_mappings_basic():
    """Test basic functionality of apply_mappings."""
    df = pd.DataFrame({
        "Job": ["A171", "A172", "A173"],
        "Housing": ["A151", "A152", "A153"]
    })
    mappings = {
        "Job": {"A171": "Unemployed", "A172": "Unskilled", "A173": "Skilled"},
        "Housing": {"A151": "Rent", "A152": "Own", "A153": "Free"}
    }
    result = apply_mappings(df, mappings)
    assert result.loc[0, "Job"] == "Unemployed"
    assert result.loc[1, "Housing"] == "Own"

def test_apply_value_mappings_missing_column():
    """Test apply_mappings with mapping for a column not in DataFrame."""
    df = pd.DataFrame({"Job": ["A171", "A172"]})
    mappings = {"MissingCol": {"X": "Y"}}
    result = apply_mappings(df, mappings)
    assert result.equals(df)

def test_apply_value_mappings_empty_df():
    """Test apply_mappings with an empty DataFrame."""
    df = pd.DataFrame()
    mappings = {"Any": {"A": "B"}}
    result = apply_mappings(df, mappings)
    assert result.empty

def test_apply_mappings_invalid_type():
    """Test apply_mappings with non-DataFrame input."""
    with pytest.raises(AttributeError):
        apply_mappings("not a dataframe", {"A": {"B": "C"}})

def test_apply_mappings_unmapped_values():
    """Test that values not in mapping dict remain unchanged."""
    df = pd.DataFrame({"Col": ["X", "Y", "Z"]})
    mappings = {"Col": {"X": "MappedX"}}  # "Y" and "Z" are unmapped
    result = apply_mappings(df, mappings)
    assert result.loc[0, "Col"] == "MappedX"
    assert result.loc[1, "Col"] == "Y"
    assert result.loc[2, "Col"] == "Z"

# Test create_output_dir

def test_create_output_dir(tmp_path):
    """Test that create_output_dir creates the specified path."""
    test_path = tmp_path / "subdir"
    create_output_dir(str(test_path))
    assert test_path.exists()

# Tests for evaluate_model

def test_evaluate_model_success(test_data, temp_output_dir):
    """Test evaluate_model returns expected metrics and creates output files."""
    X_train, X_test, y_train, y_test, _ = test_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test, "TestModel", temp_output_dir)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1', 'model_name'])
    assert metrics['model_name'] == "TestModel"
    assert os.path.exists(os.path.join(temp_output_dir, "testmodel_confusion_matrix.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "testmodel_classification_report.csv"))

def test_evaluate_model_with_scaled_data(test_data, temp_output_dir):
    """Test evaluate_model works with scaled test data."""
    X_train, X_test, y_train, y_test, _ = test_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    X_test_scaled = X_test * 2
    metrics = evaluate_model(model, X_test, y_test, "ScaledModel", temp_output_dir, X_test_scaled=X_test_scaled)
    assert isinstance(metrics, dict)
    assert metrics['model_name'] == "ScaledModel"

def test_evaluate_model_invalid_model(test_data, temp_output_dir):
    """Test evaluate_model raises error for invalid model without predict()."""
    class DummyModel: pass
    X_train, X_test, y_train, y_test, _ = test_data
    with pytest.raises(ValueError):
        evaluate_model(DummyModel(), X_test, y_test, "InvalidModel", temp_output_dir)

def test_evaluate_model_none_input(test_data, temp_output_dir):
    """Test evaluate_model raises error when X_test or y_test is None."""
    _, _, _, y_test, _ = test_data
    model = RandomForestClassifier()
    model.fit(np.zeros((2, 2)), [0, 1])  # minimal dummy fit

    with pytest.raises(ValueError):
        evaluate_model(model, None, y_test, "NoneInputTest", temp_output_dir)

# Tests for plot_feature_importance

def test_plot_feature_importance_success(test_data, temp_output_dir):
    """Test plot_feature_importance returns DataFrame and creates files."""
    X_train, X_test, y_train, y_test, feature_names = test_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    df = plot_feature_importance(model, feature_names, temp_output_dir)
    assert isinstance(df, pd.DataFrame)
    assert "Feature" in df.columns
    assert "Importance" in df.columns
    assert os.path.exists(os.path.join(temp_output_dir, "feature_importance.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "feature_importance.csv"))

def test_plot_feature_importance_error(test_data, temp_output_dir):
    """Test plot_feature_importance raises error for unsupported model."""
    X_train, X_test, y_train, y_test, feature_names = test_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    with pytest.raises(AttributeError):
        plot_feature_importance(model, feature_names, temp_output_dir)

def test_zero_feature_importance_plot(temp_output_dir):
    """Test plot_feature_importance handles zero-importance features."""
    class FakeModel:
        feature_importances_ = np.zeros(5)
    feature_names = [f"feature{i}" for i in range(5)]
    df = plot_feature_importance(FakeModel(), feature_names, temp_output_dir)
    assert df["Importance"].sum() == 0

def test_plot_feature_importance_length_mismatch(temp_output_dir):
    """Test that feature_importance fails when feature name list doesn't match model length."""
    class FakeModel:
        feature_importances_ = np.array([0.1, 0.9])

    feature_names = ['only_one_name']  # mismatch length

    with pytest.raises(ValueError):
        plot_feature_importance(FakeModel(), feature_names, temp_output_dir)

# Tests for compare_models

def test_compare_models_success(test_metrics, temp_output_dir):
    """Test compare_models returns DataFrame and saves output."""
    df = compare_models(test_metrics, temp_output_dir)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 4)
    assert all(col in df.columns for col in ['accuracy', 'precision', 'recall', 'f1'])
    assert os.path.exists(os.path.join(temp_output_dir, "model_comparison.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "model_comparison.csv"))

def test_compare_models_empty(temp_output_dir):
    """Test compare_models raises ValueError on empty input."""
    with pytest.raises(ValueError):
        compare_models([], temp_output_dir)

def test_compare_models_partial_metrics(temp_output_dir):
    """Test compare_models works even if some models have missing keys."""
    metrics = [
        {"model_name": "Model1", "accuracy": 0.8},
        {"model_name": "Model2", "precision": 0.9, "recall": 0.88}
    ]
    df = compare_models(metrics, temp_output_dir)
    assert "accuracy" in df.columns or "precision" in df.columns
    assert df.shape[0] == 2