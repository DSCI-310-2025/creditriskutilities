---
title: "creditriskutils: A Tutorial for Credit Risk Modeling"
author: "Stallon Pinto"
format: html
toc: true
number-sections: true
editor: visual
jupyter: python3
---

# Introduction

`creditriskutils` is a Python package that helps streamline **credit risk modeling** by providing reusable tools for data cleaning, model evaluation, and visualization. It's particularly useful when working with datasets that include coded categorical variables and binary credit classifications (e.g., *Good* vs. *Bad* credit).

This tutorial walks you through installing the package, applying it to a real-world dataset, and generating evaluation outputs.

---

# Installation

```{.bash}
pip install --index-url https://test.pypi.org/simple/ creditriskutils