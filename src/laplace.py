# laplace.py
# Implements the Laplace approximation for Bayesian logistic regression

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.utils import predict, compute_average_ll

def laplace_approximation(X_tilde, y, w_map):
    """Computes the Laplace approximation for Bayesian logistic regression."""
    N, D = X_tilde.shape
    
    # Compute Hessian matrix
    p = predict(X_tilde, w_map)
    W = np.diag(p * (1 - p))  # Diagonal matrix with variances
    H = -(X_tilde.T @ W @ X_tilde)  # Hessian approximation
    
    # Compute covariance matrix
    S_N_inv = -H  # Negative Hessian is the precision matrix
    S_N = np.linalg.inv(S_N_inv)
    
    return S_N