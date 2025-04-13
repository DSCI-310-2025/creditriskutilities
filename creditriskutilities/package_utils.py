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
            df_copy[col] = df_copy[col].map(mapping)
    return df_copy

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

# from model_utils.py
def evaluate_model(model, X_test, y_test, model_name, output_dir, X_test_scaled=None):
    """
    Evaluate a machine learning model and save performance metrics and visualizations.
    
    Parameters
    ----------
    model : sklearn estimator
        The trained model to evaluate
    X_test : pandas.DataFrame or numpy.ndarray
        Test features
    y_test : pandas.Series or numpy.ndarray
        True target values
    model_name : str
        Name of the model for saving files
    output_dir : str
        Directory to save evaluation results
    X_test_scaled : pandas.DataFrame or numpy.ndarray, optional
        Scaled test features, used if model requires scaled input
        
    Returns
    -------
    dict
        Dictionary containing model performance metrics
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> metrics = evaluate_model(model, X_test, y_test, "RandomForest", "./results")
    >>> metrics['accuracy'] > 0.7
    True
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use scaled data if provided
    X_test_eval = X_test_scaled if X_test_scaled is not None else X_test
    
    # Make predictions
    y_pred = model.predict(X_test_eval)
    y_pred_proba = model.predict_proba(X_test_eval)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save classification report
    report = classification_report(y_test, y_pred, target_names=['Good Credit', 'Bad Credit'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_classification_report.csv'))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good Credit', 'Bad Credit'],
                yticklabels=['Good Credit', 'Bad Credit'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'), dpi=300)
    plt.close()
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

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
        List of dictionaries containing model metrics from evaluate_model function
    output_dir : str
        Directory to save comparison results
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing model comparison metrics
        
    Raises
    ------
    ValueError
        If model_metrics_list is empty
    """
    # Check if the list is empty
    if not model_metrics_list:
        raise ValueError("Model metrics list cannot be empty")
        
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison DataFrame
    models_comparison = pd.DataFrame(model_metrics_list)
    models_comparison = models_comparison.set_index('model_name')
    
    # Save model comparison
    models_comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    
    # Plot model comparison
    plt.figure(figsize=(12, 6))
    models_comparison[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar', colormap='viridis')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    return models_comparison