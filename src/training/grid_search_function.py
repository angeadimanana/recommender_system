import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time
from datetime import datetime

from src.training.train_parallelize import train

from src.training.train_with_features import train_with_features

def grid_search(data_train_by_user, data_train_by_movie, data_test_by_user,
                param_grid, n_iterations=50):
    """
    Perform grid search over hyperparameters

    Input:
        param_grid: Dictionary with parameter names as keys and lists of values
                    Example: {
                        'k': [5, 10, 20],
                        'lamda': [0.1, 1.0, 10.0],
                        'gamma': [0.01, 0.1],
                        'tau': [0.01, 0.1, 1.0]
                    }
        n_iterations: Number of training iterations
        metric: Metric to optimize ('test_rmse' or 'test_loss')

    Output:
        DataFrame with all results
    """
    results = []
    best_params = None
    best_score = float('inf')

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    total_combinations = len(param_combinations)
    print(f"\nTotal combinations to test: {total_combinations}")
    print(f"Parameters: {param_names}")

    start_time = time.time()

    for idx, params in enumerate(param_combinations, 1):
        k = params[0]
        lamda = params[1]
        gamma = params[2]
        tau = params[3]

        print(f"[{idx}/{total_combinations}]")
        print("Testing")


        # Train model
        res= train(
          data_train_by_user,
          data_train_by_movie,
          data_test_by_user,
          k, lamda, gamma, tau, n_iterations
        )
        
        plt.figure(figsize=(6,4))
        plt.plot(res['rmse_train'], label="Train RMSE")
        plt.plot(res['rmse_test'], label="Test RMSE")
        plt.title(f"K={k}, gamma={gamma}, tau={tau}")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
        result = {
            'k': k,
            'lamda': lamda,
            'gamma': gamma,
            'tau': tau,
            'iterations_done': res['iterations_done'],
            'train_rmse_final': res['rmse_train'][-1],
            'test_rmse_final': res['rmse_test'][-1],
            'best_test_rmse': min(res['rmse_test']),
            'best_test_iter': np.argmin(res['rmse_test']) + 1,
            'overfitting_final': res['rmse_test'][-1] - res['rmse_train'][-1],
            'overfitting_at_best': res['rmse_test'][np.argmin(res['rmse_test'])] - res['rmse_train'][np.argmin(res['rmse_test'])],
            'rmse_degradation': res['rmse_test'][-1] - min(res['rmse_test'])
        }

        results.append(result)

        if result['rmse_degradation'] < 0.01:  
            if result['best_test_rmse'] < best_score:
                best_score = result['best_test_rmse']
                best_params = params
                print(f"New best stable model - RMSE: {best_score:.4f}, "
                    f"Degradation: {result['rmse_degradation']:.4f}")

    total_time = time.time() - start_time
    print(f"Grid Search completed in {total_time:.2f}s")
    print(f"Best parameters: {best_params}")

    return pd.DataFrame(results)

#for adding features case
def grid_search_features(data_train_by_user, data_train_by_movie, data_test_by_user,
                param_grid, F, F_n, n_iterations=50):
    """
    Perform grid search over hyperparameters

    Input:
        param_grid: Dictionary with parameter names as keys and lists of values
                    Example: {
                        'k': [5, 10, 20],
                        'lamda': [0.1, 1.0, 10.0],
                        'gamma': [0.01, 0.1],
                        'tau': [0.01, 0.1, 1.0]
                    }
        n_iterations: Number of training iterations
        metric: Metric to optimize ('test_rmse' or 'test_loss')

    Output:
        DataFrame with all results
    """
    results = []
    best_params = None
    best_score = float('inf')

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    total_combinations = len(param_combinations)
    print(f"\nTotal combinations to test: {total_combinations}")
    print(f"Parameters: {param_names}")
   

    start_time = time.time()

    for idx, params in enumerate(param_combinations, 1):
        k = params[0]
        lamda = params[1]
        gamma = params[2]
        tau = params[3]

        print(f"\n[{idx}/{total_combinations}]")
        print("Testing")


        # Train model
        res = train_with_features(data_train_by_user,
                                          data_train_by_movie,
                                          data_test_by_user,
                                          F, F_n, k, lamda,
                                          gamma, tau, n_iterations)

        result = {
            'k': k,
            'lamda': lamda,
            'gamma': gamma,
            'tau': tau,
            'iterations_done': res['iterations_done'],
            'train_rmse_final': res['rmse_train'][-1],
            'test_rmse_final': res['rmse_test'][-1],
            'best_test_rmse': min(res['rmse_test']),
            'best_test_iter': np.argmin(res['rmse_test']) + 1,
            'overfitting_final': res['rmse_test'][-1] - res['rmse_train'][-1],
            'overfitting_at_best': res['rmse_test'][np.argmin(res['rmse_test'])] - res['rmse_train'][np.argmin(res['rmse_test'])],
            'rmse_degradation': res['rmse_test'][-1] - min(res['rmse_test'])
            }


        results.append(result)

        if result['rmse_degradation'] < 0.01:  
            if result['best_test_rmse'] < best_score:
                best_score = result['best_test_rmse']
                best_params = params
                print(f"New best stable model - RMSE: {best_score:.4f}, "
                    f"Degradation: {result['rmse_degradation']:.4f}")


    total_time = time.time() - start_time
    print(f"Grid Search completed in {total_time:.2f}s")
    print(f"Best parameters: {best_params}")

    return pd.DataFrame(results)

