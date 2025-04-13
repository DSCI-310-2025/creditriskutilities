"""
Utility functions for loading data and producing visualizations for EDA.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# from data_cleaning.py
def apply_mappings(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Apply categorical value mappings to a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing columns with coded categorical values.

    mappings : dict
        A dictionary where keys are column names in `df`, and values are
        dictionaries mapping old values to new ones.

    Returns:
    --------
    pd.DataFrame
        A copy of the DataFrame with the specified mappings applied.

    Example:
    --------
    >>> df = pd.DataFrame({"Job": ["A171", "A172"]})
    >>> apply_mappings(df, {"Job": {"A171": "Unemployed", "A172": "Skilled"}})
           Job
    0  Unemployed
    1     Skilled
    """
    df_copy = df.copy()
    for col, mapping in mappings.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map(mapping).fillna(df_copy[col])
    return df_copy


def load_and_prepare_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw German credit data and returns a cleaned DataFrame with assigned column names.

    Parameters
    ----------
    filepath : str
        The path to the raw data file.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with proper column names.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pd.errors.ParserError
        If the file cannot be parsed.
    ValueError
        If the number of columns in the data does not match the expected count.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at path: {filepath}")

    try:
        df = pd.read_csv(filepath, sep=" ", header=None)
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse file: {e}")

    # Define expected column names (update as needed)
    column_names = [
        "Checking_Acc_Status", "Duration (in months)", "Credit_History", "Purpose",
        "Credit_Amount", "Savings_Acc", "Employment", "Installment_Rate",
        "Personal_Status", "Other_Debtors", "Residence_Since", "Property",
        "Age", "Other_Installment", "Housing", "Existing_Credits",
        "Job", "Num_People_Maintained", "Telephone", "Foreign_Worker", "Credit Standing"
    ]

    if len(df.columns) != len(column_names):
        raise ValueError("Number of columns in raw data does not match expected column names")

    df.columns = column_names
    return df


# from visualization.py
def create_output_dir(path: str):
    """
    Create the specified output directory if it does not already exist.

    Parameters
    ----------
    path : str
        Path of the directory to create.

    Notes
    -----
    This function does not raise an error if the directory already exists.
    """
    os.makedirs(path, exist_ok=True)


def plot_corr_barplot(df: pd.DataFrame, target: str, save_path: str):
    """
    Plots a bar chart of correlation coefficients with respect to the target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    target : str
        The target column to compute correlations for.
    save_path : str
        File path to save the plot image.
    
    Raises
    ------
    KeyError
        If target column is not found in df.
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot plot correlations.")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")

    corr = df.corr()[target].sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    bars = sns.barplot(x=corr.values, y=corr.index)
    plt.title(f'Feature Correlation with {target}', fontsize=16, fontweight='bold')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    for i, val in enumerate(corr.values):
        color = '#e74c3c' if val > 0 else '#3498db'
        bars.patches[i].set_color(color)
        plt.text(val + 0.01 if val >= 0 else val - 0.06, i, f'{val:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_credit_standing_distribution(df: pd.DataFrame, save_path: str):
    """
    Plots the distribution of the 'Credit Standing' variable as a bar chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    save_path : str
        File path to save the plot image.
    
    Raises
    ------
    KeyError
        If 'Credit Standing' is not in the DataFrame.
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot plot credit standing distribution.")
    if 'Credit Standing' not in df.columns:
        raise KeyError("Column 'Credit Standing' not found in DataFrame")
        
    custom_palette = ['#3498db', '#e74c3c']
    counts = df['Credit Standing'].value_counts()

    plt.figure(figsize=(8, 8))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=custom_palette)
    plt.title('Credit Standing Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Credit Standing (0=Good, 1=Bad)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Good (0)', 'Bad (1)'])

    total = len(df)
    for i, p in enumerate(ax.patches):
        pct = 100 * p.get_height() / total
        ax.annotate(f'{int(p.get_height())}\n({pct:.1f}%)',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, features: list, target: str, save_path: str):
    """
    Plots histograms (with an overlaid kernel density estimate) for each feature in 'features',
    colored by a target variable, and saves the plot to the specified path.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    features : list
        List of feature column names to be plotted.
    target : str
        The column name for the target variable to use as hue.
    save_path : str
        The file path where the plot image will be saved.
    
    Raises
    ------
    ValueError
        If the DataFrame is empty.
    KeyError
        If the target column or any feature is missing.
    TypeError
        If any feature column is not numeric.
    
    Returns
    -------
    None
        The function saves the plot to 'save_path'.
    """
    # Input validation
    if df.empty:
        raise ValueError("The DataFrame is empty. Cannot generate plots on an empty DataFrame.")
    
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")
    
    for feature in features:
        if feature not in df.columns:
            raise KeyError(f"Feature column '{feature}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise TypeError(f"Feature column '{feature}' must be numeric to plot a histogram.")
    
    # Utility: Create subplots.
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 4 * len(features)))
    if len(features) == 1:
        axes = [axes]
    
    # Utility: Plot a single feature's distribution.
    def plot_single_feature(ax, feature):
        sns.histplot(
            data=df,
            x=feature,
            hue=target,
            kde=True,
            palette=['#3498db', '#e74c3c'],
            alpha=0.6,
            bins=30,
            ax=ax,
            hue_order=[0, 1]
        )
        ax.set_title(f"Distribution of {feature} by {target}", fontsize=14, fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
    
    # Decompose: Iterate through features and plot each.
    for i, feature in enumerate(features):
        plot_single_feature(axes[i], feature)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



# from model_utils.py
def evaluate_model(model, X_test, y_test, model_name, output_dir, X_test_scaled=None):
    """
    Evaluate a machine learning model and save performance metrics and visualizations.
    """
    if not hasattr(model, "predict"):
        raise ValueError("Provided model must have a predict method.")

    if X_test is None or y_test is None:
        raise ValueError("Both X_test and y_test must be provided.")

    os.makedirs(output_dir, exist_ok=True)

    X_eval = X_test_scaled if X_test_scaled is not None else X_test
    y_pred = model.predict(X_eval)

    # Compute metrics
    metrics = _compute_classification_metrics(y_test, y_pred)
    metrics['model_name'] = model_name

    # Print metrics
    print(f"\n{model_name} Performance:")
    for k, v in metrics.items():
        if k != 'model_name':
            print(f"{k.capitalize()}: {v:.4f}")

    # Save outputs
    _save_classification_report(y_test, y_pred, model_name, output_dir)
    _plot_confusion_matrix(y_test, y_pred, model_name, output_dir)

    return metrics
    
    

def plot_feature_importance(model, feature_names, output_dir, n_top=15):
    """
    Plot and save feature importance for tree-based models.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list or array-like
        Names of the features
    output_dir : str
        Directory to save the plot
    n_top : int, optional
        Number of top features to display, default is 15
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing feature importance values
        
    Raises
    ------
    AttributeError
        If model doesn't have feature_importances_ attribute
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=5, random_state=42)
    >>> feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    >>> model = RandomForestClassifier().fit(X, y)
    >>> importance_df = plot_feature_importance(model, feature_names, "./results")
    >>> len(importance_df) == 5
    True
    """
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("Model does not have feature_importances_ attribute")
    
    if len(model.feature_importances_) != len(feature_names):
        raise ValueError("Number of feature names must match the number of feature importances.")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(n_top))
    plt.title(f'Top {n_top} Features by Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    return feature_importance

def compare_models(model_metrics_list, output_dir):
    """
    Compare multiple models and visualize their performance metrics.

    Parameters
    ----------
    model_metrics_list : list of dict
        List of dictionaries containing model metrics from evaluate_model function.
    output_dir : str
        Directory to save comparison results.

    Returns
    -------
    pd.DataFrame
        DataFrame containing model comparison metrics.

    Raises
    ------
    ValueError
        If model_metrics_list is empty.
    """
    if not model_metrics_list:
        raise ValueError("Model metrics list cannot be empty.")

    os.makedirs(output_dir, exist_ok=True)

    # Create comparison DataFrame
    models_comparison = pd.DataFrame(model_metrics_list)
    models_comparison = models_comparison.set_index('model_name')

    # Save comparison CSV
    models_comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'))

    # Dynamically select columns to plot
    plot_columns = [col for col in ['accuracy', 'precision', 'recall', 'f1'] if col in models_comparison.columns]

    # Plot
    plt.figure(figsize=(12, 6))
    models_comparison[plot_columns].plot(kind='bar', colormap='viridis')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

    return models_comparison

#helper functions
def _compute_classification_metrics(y_true, y_pred):
    """Compute basic classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }

def _save_classification_report(y_true, y_pred, model_name, output_dir):
    """Generate and save classification report to CSV."""
    report = classification_report(
        y_true, y_pred,
        target_names=['Good Credit', 'Bad Credit'],
        output_dict=True
    )
    df = pd.DataFrame(report).transpose()
    filename = f"{model_name.lower().replace(' ', '_')}_classification_report.csv"
    df.to_csv(os.path.join(output_dir, filename))

def _plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Good Credit', 'Bad Credit'],
                yticklabels=['Good Credit', 'Bad Credit'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()