# optimize.py
# Implements MAP estimation for logistic regression using L-BFGS optimization

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.utils import logistic, get_x_tilde

def negative_log_posterior(w, X_tilde, y, sigma2_0):
    """Computes the negative log-posterior for MAP estimation."""
    predictions = logistic(X_tilde @ w)
    log_likelihood = np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    log_prior = -0.5 * np.sum(w**2) / sigma2_0  # Gaussian prior
    return - (log_likelihood + log_prior)

def grad_negative_log_posterior(w, X_tilde, y, sigma2_0):
    """Computes the gradient of the negative log-posterior."""
    predictions = logistic(X_tilde @ w)
    gradient = X_tilde.T @ (y - predictions) - w / sigma2_0
    return -gradient

def compute_map_solution(X_tilde, y, sigma2_0=1.0):
    """Finds the MAP estimate using L-BFGS optimization."""
    D = X_tilde.shape[1]
    w_init = np.zeros(D)
    w_map, _, _ = fmin_l_bfgs_b(
        negative_log_posterior, w_init,
        fprime=grad_negative_log_posterior,
        args=(X_tilde, y, sigma2_0), disp=False
    )
    return w_map