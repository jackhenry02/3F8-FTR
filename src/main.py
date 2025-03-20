# main.py
# Entry point for training and evaluation

import numpy as np
import matplotlib.pyplot as plt
from src.utils import logistic, get_x_tilde, compute_average_ll
from src.laplace import laplace_approximation
from src.optimize import compute_map_solution
from src.data_loader import load_data

def run_experiment():
    """Main function to load data, train model, and evaluate results."""
    # Load dataset
    X_train, y_train, X_test, y_test = load_data()

    # Expand input features
    X_tilde_train = get_x_tilde(X_train)
    X_tilde_test = get_x_tilde(X_test)

    # Compute MAP solution
    w_map = compute_map_solution(X_tilde_train, y_train)
    print(f"MAP solution: {w_map}")

    # Compute Laplace approximation
    laplace_result = laplace_approximation(X_tilde_train, y_train, w_map)
    print(f"Laplace approximation result: {laplace_result}")

    # Compute log-likelihood
    ll_train = compute_average_ll(X_tilde_train, y_train, w_map)
    ll_test = compute_average_ll(X_tilde_test, y_test, w_map)
    print(f"Log-likelihood (Train): {ll_train}, (Test): {ll_test}")

    # Plot decision boundary
    plot_predictive_distribution(X_train, y_train, w_map)
    plot_predictive_distribution(X_test, y_test, w_map)

def plot_predictive_distribution(X, y, w):
    """Function to plot predictive probabilities of the logistic classifier."""
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    X_tilde = get_x_tilde(np.c_[xx.ravel(), yy.ravel()])
    Z = logistic(X_tilde @ w).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 10), cmap="RdBu", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap="RdBu")
    plt.title("Predictive Distribution")
    plt.show()

if __name__ == "__main__":
    run_experiment()
