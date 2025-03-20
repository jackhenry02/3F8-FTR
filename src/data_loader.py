# data_loader.py
# Handles data loading and preprocessing

import numpy as np

def load_data():
    """Loads and preprocesses the dataset."""
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')
    
    # Shuffle data
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation, :]
    y = y[permutation]
    
    # Split into training and test sets
    n_train = 800
    X_train, X_test = X[:n_train, :], X[n_train:, :]
    y_train, y_test = y[:n_train], y[n_train:]
    
    return X_train, y_train, X_test, y_test