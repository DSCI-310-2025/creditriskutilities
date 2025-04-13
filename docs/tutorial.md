---
title: "creditriskutilities: A Tutorial"
format: html
author: "Pipeline Pythons (Ayush, Stallon, Zhanerke)"
editor: visual
jupyter: python3
---

## 🧭 Introduction

This tutorial provides a hands-on walkthrough of how to use the `creditriskutilities` package for evaluating machine learning models in a credit risk analysis pipeline.

We will cover:

- Installing the package
- Applying value mappings
- Evaluating a classification model
- Plotting feature importance
- Comparing multiple models

The tutorial assumes you are working with binary classification models (e.g., predicting loan default) and have some familiarity with `pandas` and `scikit-learn`.

---

## Installation

To install the package from [TestPyPI](https://test.pypi.org/project/creditriskutilities/):

```bash
pip install --index-url https://test.pypi.org/simple/ creditriskutilities
```

---

## Imports and Setup

```python

# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from creditriskutilities import (
    apply_mappings,
    create_output_dir,
    evaluate_model,
    plot_feature_importance,
    compare_models
    )

```

---

## Generate Simulated Data

```python

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=5, random_state=42)
feature_names = [f"feature{i}" for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

```

---

###  create_output_dir()

```python

create_output_dir("results")

```
### evaluate_model()

```python
metrics = evaluate_model(model, X_test, y_test, "RandomForest", "results")
metrics

```

### plot_feature_importance()

```python

plot_feature_importance(model, feature_names, "results")

```

### compare_models()

```python

metrics_1 = {'model_name': 'Model 1', 'accuracy': 0.81, 'precision': 0.75, 'recall': 0.70, 'f1': 0.72}
metrics_2 = {'model_name': 'Model 2', 'accuracy': 0.85, 'precision': 0.80, 'recall': 0.78, 'f1': 0.79}

comparison_df = compare_models([metrics_1, metrics_2], "results")
comparison_df

```

### apply_mappings()

```python

df = pd.DataFrame({
    "Job": ["A171", "A172", "A173"],
    "Housing": ["A151", "A152", "A153"]
})

mappings = {
    "Job": {
        "A171": "Unemployed",
        "A172": "Unskilled",
        "A173": "Skilled"
    },
    "Housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "Free"
    }
}

apply_mappings(df, mappings)

```