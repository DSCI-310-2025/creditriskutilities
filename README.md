# creditriskutilities

[![CI](https://github.com/DSCI-310-2025/creditriskutils/actions/workflows/ci.yml/badge.svg)](https://github.com/DSCI-310-2025/creditriskutils/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DSCI-310-2025/creditriskutils/branch/main/graph/badge.svg)](https://codecov.io/gh/DSCI-310-2025/creditriskutils)

`creditriskutilities` is a lightweight Python package developed to streamline the evaluation and comparison of credit risk classification models. The package abstracts commonly repeated tasks in model assessment — such as computing classification metrics, visualizing feature importance, applying value mappings, and generating side-by-side model comparisons — into standalone, reusable functions. These functions are especially useful in credit analytics pipelines, where consistent and interpretable model evaluation is critical.

---

## Why This Package?

In credit risk analytics, we often work with:
- Encoded categorical data
- Tree-based models with feature importances
- A need to report multiple evaluation metrics
- Repetitive output saving and visualization logic

Instead of rewriting code across notebooks and scripts, `creditriskutilities` offers clean, modular tools you can plug directly into your pipeline.

---

## Key Features

- **evaluate_model**: Automatically compute and save accuracy, precision, recall, and F1 — plus a CSV report and confusion matrix plot.
- **plot_feature_importance**: Visualize top N features by importance and save plot + CSV.
- **compare_models**: Compare multiple model metrics in a single bar plot.
- **apply_mappings**: Map coded values to readable labels using a dictionary — great for pre-modeling pipelines.
- **create_output_dir**: Safely create folders as needed (used internally by other functions).

---

## Who Should Use This?

This package is ideal for:
- Students and teams building machine learning pipelines in finance
- Practitioners working with classification problems (especially in credit scoring)
- Developers who want to avoid re-writing evaluation logic in every notebook or script

---

### Authors

- Stallon Pinto
- Ayush Joshi
- Zhanerke Zhumash
