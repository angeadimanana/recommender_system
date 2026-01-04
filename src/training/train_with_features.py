
import numpy as np
import time
from numba import njit, prange

from src.data_struct.data_structure import convert_structure


@njit(parallel=True, fastmath=True)
def update_user_biases_and_vectors_with_features(indptr, indices, values,
                                    u_biases, m_biases, u, v,
                                    lamda, gamma, tau):
    """Update user latent vectors and biases using ALS.

        Input:
            indptr, array:  index pointer for user ratings.
            indices, array: movie indices for ratings.
            values, array:  containing the rating values.
            u_biases, array: user biases
            m_biases, array:  movie biases.
            u, array: user embedding  matrix (m x k) 
            v, array: movie embedding matrix (n x k) 
            lamda, gamma, tau (float): parameters 
           
        Output:
            u: updated user embedding latent matrix.
            u_biases: updated user bias array.
    """
    m = len(u_biases)
    k = u.shape[1]

    for i in prange(m):
        start = indptr[i]
        end = indptr[i + 1]
        n_ratings = end - start
        if n_ratings == 0:
            continue

        idxs = indices[start:end]
        ratings = values[start:end]
        V_subset = v[idxs, :]

        # Update bias
        prod_scal = V_subset @ u[i]
        res = ratings - prod_scal - m_biases[idxs]
        bias = lamda * np.sum(res)
        u_biases[i] = bias / (lamda * n_ratings + gamma)

        # Update vector
        val = ratings - u_biases[i] - m_biases[idxs]
        s_1 = lamda * (V_subset.T @ val)
        s_2 = tau * np.eye(k) + lamda * (V_subset.T @ V_subset)
        u[i] = np.linalg.solve(s_2, s_1)

    return u, u_biases


@njit(parallel=True, fastmath=True)
def update_movie_biases_and_vectors_with_features(indptr, indices, values,
                                                   m_biases, u_biases, v, u,
                                                   F, F_n, f_vectors,
                                                   lamda, gamma, tau):
    """
    Update movie biases and vectors with feature priors

  
    """
    n = len(m_biases)
    k = v.shape[1]

    for j in prange(n):
        start = indptr[j]
        end = indptr[j + 1]
        n_ratings = end - start

        # Update bias 
        if n_ratings > 0:
            idxs = indices[start:end]
            ratings = values[start:end]
            U_subset = u[idxs, :]

            prod_scal = U_subset @ v[j]
            res = ratings - prod_scal - u_biases[idxs]
            bias = lamda * np.sum(res)
            m_biases[j] = bias / (lamda * n_ratings + gamma)

        # Update vector with feature prior
        if n_ratings > 0:
            idxs = indices[start:end]
            ratings = values[start:end]
            U_subset = u[idxs, :]

            val = ratings - m_biases[j] - u_biases[idxs]
            s_1 = lamda * (U_subset.T @ val)

            num_features = F_n[j]
            if num_features > 0:
                feature_sum = np.zeros(k)
                for l in range(f_vectors.shape[0]):
                    if F[j, l] > 0:  
                        feature_sum += f_vectors[l]

                s_1 += (tau / np.sqrt(float(num_features))) * feature_sum

            s_2 = tau * np.eye(k) + lamda * (U_subset.T @ U_subset)
            v[j] = np.linalg.solve(s_2, s_1)

    return v, m_biases


@njit(parallel=False)  
def update_feature_vectors(v, F, F_n, f_vectors):
    """
    Update feature vectors

    Input:
        v: movie embedding latent matrix (n x k).
        F (array): binary matrix indicating which movies have which features.
        F_n (array): number of features associated with each movie.
        f_vectors (array): feature latent vectors 
        tau (float): hyperparameter

    Output:
        f_vectors: updated feature latent matrix (num_features x k).
    """
    num_features = f_vectors.shape[0]
    num_movies = F.shape[0]
    k = f_vectors.shape[1]

    for i in range(num_features):
        numerator = np.zeros(k)
        denominator = 0.0

        for n in range(num_movies):
            if F[n, i] > 0 and F_n[n] > 0:
                sqrt_Fn = np.sqrt(float(F_n[n]))
                weight = 1.0 / sqrt_Fn

                other_features_sum = np.zeros(k)
                for l in range(num_features):
                    if l != i and F[n, l] > 0:
                        other_features_sum += f_vectors[l]

                term = v[n] - (other_features_sum / sqrt_Fn)

                numerator += weight * term
                denominator += weight**2

        if denominator > 0:
            f_vectors[i] = numerator / (denominator + 1)

    return f_vectors


@njit(parallel=True, fastmath=True)
def cost_function_with_features(indptr, indices, values,
                                 u_biases, m_biases, u, v,
                                 F, F_n, f_vectors,
                                 lamda, gamma, tau):
    """
    Compute loss with features 

    Input:
        indptr, indices, values:  data structure for rating data
        u_biases, m_biases (array): user and movie bias
        u, v (array): user and movie latent matrices
        F, F_n (array): Feature assignment matrix and feature counts.
        f_vectors (array): features embedding latent vectors.
        lamda, gamma, tau (float): Model hyperparameters.

    Output:
         (rmse, loss) where rmse is the Root Mean Square Error and 
            loss is the total value of the regularized objective function.
    
    """
    m = len(u_biases)
    total_squared_error = 0.0
    count = 0

    # Compute prediction error
    for i in prange(m):
        start = indptr[i]
        end = indptr[i + 1]
        n_ratings = end - start
        if n_ratings == 0:
            continue

        idxs = indices[start:end]
        ratings = values[start:end]
        V_subset = v[idxs, :]

        for j in range(n_ratings):
            pred = np.dot(V_subset[j, :], u[i, :]) + u_biases[i] + m_biases[idxs[j]]
            err = ratings[j] - pred
            total_squared_error += err * err
            count += 1

    if count == 0:
        return 0.0, 0.0

    rmse = np.sqrt(total_squared_error / count)

    # Bias regularization
    reg_biases = gamma * (np.sum(u_biases ** 2) + np.sum(m_biases ** 2)) / 2.0

    # User vector regularization
    reg_users = tau * np.sum(u ** 2) / 2.0

    # Movie vector regularization with feature 
    reg_movies = 0.0
    for n in range(v.shape[0]):
        if F_n[n] > 0:
            feature_mean = np.zeros(v.shape[1])
            sqrt_Fn = np.sqrt(float(F_n[n]))
            for l in range(f_vectors.shape[0]):
                if F[n, l] > 0:
                    feature_mean += f_vectors[l] / sqrt_Fn

            diff = v[n] - feature_mean
            reg_movies += np.sum(diff ** 2)
        else:
            reg_movies += np.sum(v[n] ** 2)

    reg_movies *= tau / 2.0

    loss = (lamda * total_squared_error / 2.0) + reg_biases + reg_users + reg_movies

    return rmse, loss


def train_with_features(data_train_by_user, data_train_by_movie,
                        data_test_by_user, F, F_n,
                        k, lamda, gamma, tau, N):
    """
    Train recommendation system with feature priors

    Input:
        data_train_by_user, data_train_by_movie: training data
        data_test_by_user: test data
        F: binary feature matrix (num_movies x num_features)
        F_n: number of features per movie
        k: latent dimension
        lamda, gamma, tau: regularization parameters
        N: number of iterations

    Output:
        user_biases, movie_biases, u, v, f_vectors: learned parameters
        costs_train, rmse_train, costs_test, rmse_test: training history
    """
    m = len(data_train_by_user)
    n = len(data_train_by_movie)
    num_features = F.shape[1]

    # Initialize parameters
    user_biases = np.zeros(m)
    movie_biases = np.zeros(n)
    u = np.random.randn(m, k) / np.sqrt(k)
    v = np.random.randn(n, k) / np.sqrt(k)
    f_vectors = np.random.randn(num_features, k) / np.sqrt(k)

    costs_train = []
    rmse_train = []
    costs_test = []
    rmse_test = []

    print(f"Training with features:")
    print(f"  k={k}, lamda={lamda}, gamma={gamma}, tau={tau}")
    print(f"  Users: {m}, Movies: {n}, Features: {num_features}\n")

    indptr_user, indices_user, values_user = convert_structure(data_train_by_user)
    indptr_movie, indices_movie, values_movie = convert_structure(data_train_by_movie)
    indptr_test_user, indices_test_user, values_test_user = convert_structure(data_test_by_user)

    total_duration = 0
    start_time = time.time()

    for iteration in range(N):
        # Update users 
        u, user_biases = update_user_biases_and_vectors_with_features(
            indptr_user, indices_user, values_user,
            user_biases, movie_biases, u, v,
            lamda, gamma, tau
        )

        # Update movies (with feature)
        v, movie_biases = update_movie_biases_and_vectors_with_features(
            indptr_movie, indices_movie, values_movie,
            movie_biases, user_biases, v, u,
            F, F_n, f_vectors,
            lamda, gamma, tau
        )

        # Update features
        f_vectors = update_feature_vectors(v, F, F_n, f_vectors)

        
        r_train, loss_train = cost_function_with_features(
            indptr_user, indices_user, values_user,
            user_biases, movie_biases, u, v,
            F, F_n, f_vectors,
            lamda, gamma, tau
        )

        r_test, loss_test = cost_function_with_features(
            indptr_test_user, indices_test_user, values_test_user,
            user_biases, movie_biases, u, v,
            F, F_n, f_vectors,
            lamda, gamma, tau
        )

        rmse_train.append(r_train)
        costs_train.append(loss_train)
        rmse_test.append(r_test)
        costs_test.append(loss_test)

        if (iteration + 1) % 5 == 0 or iteration == 0:
            duration = time.time() - start_time
            total_duration += duration

            print(f"Iter {iteration + 1:3d}/{N}\t"
                  f"Train Loss: {loss_train:10.4f}\t"
                  f"Train RMSE: {r_train:6.4f}\t"
                  f"Test Loss: {loss_test:10.4f}\t"
                  f"Test RMSE: {r_test:6.4f}\t"
                  f"Time: {duration:6.2f}s")
            start_time = time.time()

    print(f"\nTotal training time: {total_duration:.2f}s")

    return {
        'k': k,
        'lamda': lamda,
        'gamma': gamma,
        'tau': tau,
        'user_biases': user_biases,
        'movie_biases': movie_biases,
        'u': u,
        'v': v,
        'f_vectors': f_vectors,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'costs_train': costs_train,
        'costs_test': costs_test,
        'duration': total_duration
    }