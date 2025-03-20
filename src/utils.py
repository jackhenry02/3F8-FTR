# utils.py
# Utility functions for logistic regression

import numpy as np

def logistic(x):
    """Logistic (sigmoid) function."""
    return 1.0 / (1.0 + np.exp(-x))

def get_x_tilde(X):
    """Expands a matrix of input features by adding a column of ones."""
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

def predict(X_tilde, w):
    """Makes predictions with a logistic classifier."""
    return logistic(X_tilde @ w)

def compute_average_ll(X_tilde, y, w):
    """Computes the average log-likelihood of the logistic classifier on a dataset."""
    output_prob = predict(X_tilde, w)
    return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))