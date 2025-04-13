import os
import pytest
import pandas as pd
from creditriskutilities import apply_mappings
from creditriskutilities import create_output_dir
from creditriskutilities import evaluate_model, plot_feature_importance, compare_models
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Import the functions to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_apply_value_mappings_basic():
    df = pd.DataFrame({
        "Job": ["A171", "A172", "A173"],
        "Housing": ["A151", "A152", "A153"]
    })

    mappings = {
        "Job": {
            "A171": "Unemployed / Unskilled (Non-Resident)",
            "A172": "Unskilled (Resident)",
            "A173": "Skilled Employee / Official"
        },
        "Housing": {
            "A151": "Rent",
            "A152": "Own",
            "A153": "For Free"
        }
    }

    result = apply_mappings(df, mappings)

    assert result.loc[0, "Job"] == "Unemployed / Unskilled (Non-Resident)"
    assert result.loc[1, "Housing"] == "Own"

def test_apply_value_mappings_missing_column():
    df = pd.DataFrame({
        "Job": ["A171", "A172"]
    })
    mappings = {
        "MissingCol": {"X": "Y"}
    }
    result = apply_mappings(df, mappings)
    assert result.equals(df)

def test_apply_value_mappings_empty_df():
    df = pd.DataFrame()
    mappings = {
        "Any": {"A": "B"}
    }
    result = apply_mappings(df, mappings)
    assert result.empty

def test_create_output_dir(tmp_path):
    test_path = tmp_path / "subdir"
    create_output_dir(str(test_path))
    assert test_path.exists()


# Create test data fixtures
@pytest.fixture
def temp_output_dir(tmpdir):
    """Create a temporary directory for test outputs."""
    return str(tmpdir.mkdir("test_output"))

@pytest.fixture
def test_data():
    """Create test data for model evaluation."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate classification data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    feature_names = [f'feature{i}' for i in range(5)]
    
    return X_train, X_test, y_train, y_test, feature_names

@pytest.fixture
def test_metrics():
    """Create test metrics for model comparison."""
    metrics1 = {'model_name': 'Model1', 'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6, 'f1': 0.65}
    metrics2 = {'model_name': 'Model2', 'accuracy': 0.9, 'precision': 0.85, 'recall': 0.8, 'f1': 0.82}
    return [metrics1, metrics2]

# Test evaluate_model function
def test_evaluate_model_success(test_data, temp_output_dir):
    """Test that evaluate_model returns correct metrics and creates expected files."""
    X_train, X_test, y_train, y_test, _ = test_data
    
    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Test the function
    metrics = evaluate_model(model, X_test, y_test, "TestModel", temp_output_dir)
    
    # Check if metrics are returned correctly
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert metrics['model_name'] == "TestModel"
    
    # Check if files were created
    assert os.path.exists(os.path.join(temp_output_dir, "testmodel_confusion_matrix.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "testmodel_classification_report.csv"))

def test_evaluate_model_with_scaled_data(test_data, temp_output_dir):
    """Test that evaluate_model works correctly with scaled data."""
    X_train, X_test, y_train, y_test, _ = test_data
    
    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Create scaled data
    X_test_scaled = X_test * 2  # Simple scaling for testing
    
    # Test the function with scaled data
    metrics = evaluate_model(model, X_test, y_test, "ScaledModel", temp_output_dir, X_test_scaled=X_test_scaled)
    
    # Check if metrics are returned correctly
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert metrics['model_name'] == "ScaledModel"

# Test plot_feature_importance function
def test_plot_feature_importance_success(test_data, temp_output_dir):
    """Test that plot_feature_importance returns correct DataFrame and creates expected files."""
    X_train, X_test, y_train, y_test, feature_names = test_data
    
    # Train a model with feature importance
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Test the function
    importance_df = plot_feature_importance(model, feature_names, temp_output_dir)
    
    # Check if DataFrame is returned correctly
    assert isinstance(importance_df, pd.DataFrame)
    assert 'Feature' in importance_df.columns
    assert 'Importance' in importance_df.columns
    assert len(importance_df) == len(feature_names)
    
    # Check if files were created
    assert os.path.exists(os.path.join(temp_output_dir, "feature_importance.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "feature_importance.csv"))

def test_plot_feature_importance_error(test_data, temp_output_dir):
    """Test that plot_feature_importance raises an error for models without feature_importances_."""
    X_train, X_test, y_train, y_test, feature_names = test_data
    
    # Train a model without feature_importances_
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Test that the function raises an error
    with pytest.raises(AttributeError):
        plot_feature_importance(model, feature_names, temp_output_dir)

# Test compare_models function
def test_compare_models_success(test_metrics, temp_output_dir):
    """Test that compare_models returns correct DataFrame and creates expected files."""
    # Test the function
    comparison = compare_models(test_metrics, temp_output_dir)
    
    # Check if DataFrame is returned correctly
    assert isinstance(comparison, pd.DataFrame)
    assert comparison.shape == (2, 4)  # 2 models, 4 metrics
    assert 'accuracy' in comparison.columns
    assert 'precision' in comparison.columns
    assert 'recall' in comparison.columns
    assert 'f1' in comparison.columns
    
    # Check if files were created
    assert os.path.exists(os.path.join(temp_output_dir, "model_comparison.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "model_comparison.csv"))

def test_compare_models_empty(temp_output_dir):
    """Test that compare_models handles empty input correctly."""
    # Test with empty list
    with pytest.raises(ValueError):
        compare_models([], temp_output_dir)