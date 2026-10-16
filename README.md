# Boston Housing Price Prediction

## Overview

This project explores house price prediction using machine learning and Python. It uses the Boston Housing dataset from OpenML to investigate the relationship between housing features and house prices.

## Models

- **Linear Regression:** A baseline model for predicting house prices.
- **Lasso Regression:** A regularized linear model that can reduce some feature coefficients to zero.

## Dataset

The dataset contains 13 features describing housing characteristics and a target representing median house values.

Source: [Boston Housing Dataset on OpenML](https://www.openml.org/search?type=data&sort=runs&id=531)

## Technologies

- Python
- NumPy
- Matplotlib
- scikit-learn

## What I Learned

- Loading and exploring a dataset
- Separating features and target values
- Splitting data into training and testing sets
- Training regression models
- Making predictions and evaluating models using R²
- Exploring regularization with Lasso Regression

## How to Run

1. Install Python.
2. Install the required libraries:

   `pip install numpy matplotlib scikit-learn`

3. Run the Python script:

   `python boston_regression.py`

## Future Improvements

- Compare Linear Regression and Lasso Regression.
- Standardize features before applying Lasso.
- Select the regularization parameter using cross-validation.
- Evaluate model generalization and prediction errors.

## Status

This is a learning project created to practice machine learning fundamentals with Python and scikit-learn.
