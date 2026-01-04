import numpy as np
from numba import njit, prange
import time

from src.data_struct.data_structure import convert_structure

def update_bias(data_train_by_user, user_biases,movie_biases,lamda, gamma):
    m = len(data_train_by_user)
    
    for i in range(m):
        bias = 0
        count = 0
        for (n,r) in data_train_by_user[i]:
            bias += lamda * (r - movie_biases[n])
            count += 1
        bias = bias / (lamda * count + gamma)
        user_biases[i] = bias
    
   


def cost_function(data_train, user_biases, movie_biases, lamda, gamma):
    m = len(data_train)

    loss_train = 0
    count = 0

    for x in range(m):
      for (n,r) in data_train[x]:
        loss_train += (r - user_biases[x] - movie_biases[n])**2
        count += 1
    
    rmse = np.sqrt(loss_train / count)
    loss_train =  lamda * loss_train /2
    loss_train = loss_train + gamma * np.sum(user_biases**2) / 2
    loss_train = loss_train + gamma * np.sum(movie_biases**2) / 2

    return rmse, loss_train

def train(data_train_by_user, data_train_by_movie, data_test_by_user, lamda, gamma, N):
    
    user_biases = np.zeros(len(data_train_by_user))
    movie_biases = np.zeros(len(data_train_by_movie))
    
    costs_train = []
    rmse_train_list = []
    costs_test = []
    rmse_test_list = []

    
    for tmp in range(N) :
        update_bias(data_train_by_user,  user_biases, movie_biases, lamda, gamma)
        
        update_bias(data_train_by_movie, movie_biases, user_biases, lamda, gamma)

        rmse_train, cout_train = cost_function(data_train_by_user, user_biases, movie_biases, lamda, gamma)
        rmse_test, cout_test  = cost_function(data_test_by_user, user_biases, movie_biases, lamda, gamma) 
        
        costs_train.append(cout_train)
        rmse_train_list.append(rmse_train)
        costs_test.append(cout_test)
        rmse_test_list.append(rmse_test)

    return user_biases, movie_biases, costs_train, rmse_train_list, rmse_test_list, costs_test


# parallelization with numba
@njit(parallel=True, fastmath=True)
def update_bias_numba(indptr, indices, values, user_biases, movie_biases, lamda, gamma):
    """Update biases using numba for speed"""
    m = len(user_biases)

    for i in prange(m):
        start = indptr[i]
        end = indptr[i + 1]
        n_ratings = end - start
        if n_ratings == 0:
            continue

        idxs = indices[start:end]
        ratings = values[start:end]

        bias = lamda * np.sum(ratings - movie_biases[idxs])
        user_biases[i] = bias / (lamda * n_ratings + gamma)

    return user_biases

@njit(parallel=True, fastmath=True)
def cost_function_numba(indptr, indices, values, user_biases, movie_biases, lamda, gamma):
    """Compute cost function for bias only model"""
    m = len(user_biases)
    total_squared_error = 0.0
    count = 0

    for i in prange(m):
        start = indptr[i]
        end = indptr[i + 1]
        n_ratings = end - start
        if n_ratings == 0:
            continue

        idxs = indices[start:end]
        ratings = values[start:end]

        for j in range(n_ratings):
            pred = user_biases[i] + movie_biases[idxs[j]]
            err = ratings[j] - pred
            total_squared_error += err * err
            count += 1

    if count == 0:
        return 0.0, 0.0

    rmse = np.sqrt(total_squared_error / count)
    reg_biases = gamma * (np.sum(user_biases ** 2) + np.sum(movie_biases ** 2)) / 2.0
    loss = (lamda * total_squared_error / 2.0) + reg_biases

    return rmse, loss

# def convert_structure(data):
#     """Convert data to CSR format"""
#     indptr = [0]
#     indices = []
#     values = []
#     for element in data:
#         for idx, rate in element:
#             indices.append(idx)
#             values.append(rate)
#         indptr.append(len(indices))
#     return np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32), np.array(values, dtype=np.float64)

def train_numba(data_train_by_user, data_train_by_movie, data_test_by_user,
                    lamda, gamma, N):
    """Train bias-only model"""
    print("\nTraining Bias Only Model...")
    print(f"  lamda={lamda}, gamma={gamma}, N={N}")

    m = len(data_train_by_user)
    n = len(data_train_by_movie)
    user_biases = np.zeros(m)
    movie_biases = np.zeros(n)

    # Convert to CSR
    indptr_user, indices_user, values_user = convert_structure(data_train_by_user)
    indptr_movie, indices_movie, values_movie = convert_structure(data_train_by_movie)
    indptr_test, indices_test, values_test = convert_structure(data_test_by_user)

    costs_train = []
    rmse_train = []
    costs_test = []
    rmse_test = []

    start_time = time.time()

    for iteration in range(N):
        # Update user biases
        user_biases = update_bias_numba(indptr_user, indices_user, values_user,
                                        user_biases, movie_biases, lamda, gamma)

        # Update movie biases
        movie_biases = update_bias_numba(indptr_movie, indices_movie, values_movie,
                                         movie_biases, user_biases, lamda, gamma)

        # Compute metrics
        r_train, loss_train = cost_function_numba(indptr_user, indices_user, values_user,
                                                      user_biases, movie_biases, lamda, gamma)
        r_test, loss_test = cost_function_numba(indptr_test, indices_test, values_test,
                                                    user_biases, movie_biases, lamda, gamma)

        rmse_train.append(r_train)
        costs_train.append(loss_train)
        rmse_test.append(r_test)
        costs_test.append(loss_test)

        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  Iter {iteration+1:3d}/{N}: Train RMSE={r_train:.4f}, Test RMSE={r_test:.4f}")

    duration = time.time() - start_time
    print(f"  Completed in {duration:.2f}s")
    print(f"  Final Train RMSE: {rmse_train[-1]:.4f}")
    print(f"  Final Test RMSE: {rmse_test[-1]:.4f}\n")

    return {
        'lamda': lamda,
        'gamma': gamma,
        'user_biases': user_biases,
        'movie_biases': movie_biases,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'costs_train': costs_train,
        'costs_test': costs_test,
        'duration': duration
    }
