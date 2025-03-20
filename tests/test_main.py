import unittest
from src.main import main

class TestMain(unittest.TestCase):
    def test_main(self):
        # Simply checking if it runs without errors
        self.assertIsNone(main())


# test_main.py
# Unit tests for the main functions

import numpy as np
import pytest
from src.main import run_experiment
from src.utils import logistic, get_x_tilde, compute_average_ll
from src.laplace import laplace_approximation
from src.optimize import compute_map_solution
from src.data_loader import load_data

def test_logistic():
    """Test logistic function output range."""
    x = np.array([-100, 0, 100])
    output = logistic(x)
    assert np.all(output >= 0) and np.all(output <= 1)

def test_get_x_tilde():
    """Test feature expansion with bias term."""
    X = np.array([[1, 2], [3, 4]])
    X_tilde = get_x_tilde(X)
    assert np.all(X_tilde[:, 0] == 1)

def test_map_solution():
    """Test MAP estimation on small synthetic dataset."""
    X_train = np.array([[0.5, 1], [1, 1.5], [2, 3]])
    y_train = np.array([0, 1, 1])
    X_tilde_train = get_x_tilde(X_train)
    w_map = compute_map_solution(X_tilde_train, y_train)
    assert w_map.shape[0] == X_tilde_train.shape[1]

def test_laplace():
    """Test Laplace approximation computation."""
    X_train = np.array([[0.5, 1], [1, 1.5], [2, 3]])
    y_train = np.array([0, 1, 1])
    X_tilde_train = get_x_tilde(X_train)
    w_map = compute_map_solution(X_tilde_train, y_train)
    S_N = laplace_approximation(X_tilde_train, y_train, w_map)
    assert S_N.shape[0] == S_N.shape[1]  # Covariance matrix should be square


if __name__ == '__main__':
    unittest.main()
