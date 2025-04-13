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
    load_and_prepare_raw_data,
    plot_corr_barplot,
    plot_credit_standing_distribution,
    plot_feature_distributions
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

@pytest.fixture
def mock_df():
    """Returns a minimal valid DataFrame for plotting tests."""
    return pd.DataFrame({
        'Credit_Amount': [1000, 2000],
        'Age': [30, 40],
        'Duration (in months)': [12, 24],
        'Credit Standing': [0, 1]
    })


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


# --- Tests for load_and_prepare_raw_data ---

def test_load_and_prepare_raw_data_valid(tmp_path):
    """Test that valid raw data is correctly loaded and column names are assigned."""
    content = "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A191 A201 1"
    file_path = tmp_path / "data.txt"
    file_path.write_text(content)
    df = load_and_prepare_raw_data(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 21)
    assert "Credit_Amount" in df.columns

def test_load_and_prepare_raw_data_missing_file():
    """Test that a FileNotFoundError is raised when the file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_and_prepare_raw_data("non_existent.csv")

def test_load_and_prepare_raw_data_wrong_format(tmp_path):
    """Test that ValueError is raised when column count doesn't match expected."""
    content = "A11 6 A34"
    path = tmp_path / "bad.txt"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_and_prepare_raw_data(str(path))


# Test create_output_dir

def test_create_output_dir(tmp_path):
    """Test that create_output_dir creates the specified path."""
    test_path = tmp_path / "subdir"
    create_output_dir(str(test_path))
    assert test_path.exists()

# --- Tests for plot_corr_barplot ---

def test_plot_corr_barplot_valid(mock_df, tmp_path):
    """Test correlation bar plot is saved successfully with valid input."""
    path = tmp_path / "corr_plot.png"
    plot_corr_barplot(mock_df, 'Credit Standing', str(path))
    assert os.path.exists(path)

def test_plot_corr_barplot_missing_target(mock_df, tmp_path):
    """Test that KeyError is raised if the target column is missing."""
    df = mock_df.drop(columns=['Credit Standing'])
    path = tmp_path / "missing.png"
    with pytest.raises(KeyError):
        plot_corr_barplot(df, 'Credit Standing', str(path))

# --- Tests for plot_credit_standing_distribution ---

def test_plot_credit_distribution_valid(mock_df, tmp_path):
    """Test credit standing distribution plot is saved correctly."""
    path = tmp_path / "dist.png"
    plot_credit_standing_distribution(mock_df, str(path))
    assert os.path.exists(path)

def test_plot_credit_distribution_missing_column(tmp_path):
    """Test that KeyError is raised if 'Credit Standing' is missing."""
    df = pd.DataFrame({'SomeCol': [1, 2]})
    path = tmp_path / "missing.png"
    with pytest.raises(KeyError):
        plot_credit_standing_distribution(df, str(path))

# --- Tests for plot_feature_distributions ---

def test_plot_feature_distributions_valid(mock_df, tmp_path):
    """Test feature distributions are plotted and saved for numeric features."""
    path = tmp_path / "features.png"
    features = ['Credit_Amount', 'Age']
    plot_feature_distributions(mock_df, features, 'Credit Standing', str(path))
    assert os.path.exists(path)

def test_plot_feature_distributions_missing_feature(mock_df, tmp_path):
    """Test KeyError is raised if a listed feature is missing in the DataFrame."""
    path = tmp_path / "bad_feature.png"
    with pytest.raises(KeyError):
        plot_feature_distributions(mock_df, ['Missing_Col'], 'Credit Standing', str(path))

def test_plot_feature_distributions_missing_target(mock_df, tmp_path):
    """Test KeyError is raised if the target column is not in the DataFrame."""
    df = mock_df.drop(columns=['Credit Standing'])
    path = tmp_path / "bad_target.png"
    with pytest.raises(KeyError):
        plot_feature_distributions(df, ['Credit_Amount'], 'Credit Standing', str(path))

def test_plot_feature_distributions_empty_df(tmp_path):
    """Test ValueError is raised for empty DataFrame input."""
    df = pd.DataFrame()
    path = tmp_path / "empty.png"
    with pytest.raises(ValueError):
        plot_feature_distributions(df, ['Credit_Amount'], 'Credit Standing', str(path))

def test_plot_feature_distributions_non_numeric(tmp_path):
    """Test TypeError is raised for non-numeric feature columns."""
    df = pd.DataFrame({
        'Credit_Amount': ['low', 'medium'],
        'Age': [30, 40],
        'Credit Standing': [0, 1]
    })
    path = tmp_path / "non_numeric.png"
    with pytest.raises(TypeError):
        plot_feature_distributions(df, ['Credit_Amount'], 'Credit Standing', str(path))


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