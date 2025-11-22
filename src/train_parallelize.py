import numpy as np
import time
from numba import njit, prange

def convert_structure(data):
    indptr = [0]
    indices = []
    values = []
    for element in data:
        for idx, rate in element:
            indices.append(idx)
            values.append(rate)
        indptr.append(len(indices))
    return np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32), np.array(values, dtype=np.float64)


@njit(parallel=True, fastmath=True)
def update_biases_n_vec_embedding(indptr, indices, values,
                                      u_biases, m_biases, u, v,
                                      lamda, gamma, tau):
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

        prod_scal = V_subset @ u[i]
        res = ratings - prod_scal - m_biases[idxs]

        bias = lamda * np.sum(res)
        u_biases[i] = bias / (lamda * n_ratings + gamma)


        val = ratings - u_biases[i] - m_biases[idxs]
        s_1 = lamda * (V_subset.T @ val)
        s_2 = tau * np.eye(k) + lamda * (V_subset.T @ V_subset)
        u[i] = np.linalg.solve(s_2, s_1)

    return u, u_biases



def regularization_vect(u, tau):
    return tau * np.sum(u**2) / 2


def regularization_biases(bias, gamma):
    return gamma * np.sum(bias**2) / 2



@njit(parallel=True, fastmath=True)
def cost_function_csr(indptr, indices, values, u_biases, m_biases, u, v, lamda, gamma, tau):
    m = len(u_biases)
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
        V_subset = v[idxs, :]

        for j in range(n_ratings):
            pred = np.dot(V_subset[j, :], u[i, :]) + u_biases[i] + m_biases[idxs[j]]
            err = ratings[j] - pred
            total_squared_error += err * err
            count += 1

    if count == 0:
        return 0.0, 0.0

    rmse = np.sqrt(total_squared_error / count)

    reg_biases = gamma * (np.sum(u_biases ** 2) + np.sum(m_biases ** 2)) / 2.0
    reg_vectors = tau * (np.sum(u ** 2) + np.sum(v ** 2)) / 2.0
    loss = (lamda * total_squared_error / 2.0) + reg_biases + reg_vectors

    return rmse, loss



def train(data_train_by_user, data_train_by_movie, data_test_by_user, 
           k, lamda, gamma, tau, N):
    m = len(data_train_by_user)
    n = len(data_train_by_movie)
    user_biases = np.zeros(m)
    movie_biases = np.zeros(n)
    u = np.random.randn(m, k) / np.sqrt(k)
    v = np.random.randn(n, k) / np.sqrt(k)

    costs_train = []
    rmse_train = []
    costs_test = []
    rmse_test = []

    print(f" k={k}")
    print(f"lamda={lamda}, gamma={gamma}, tau={tau}\n")

    total_duration = 0
    start_time = time.time()

    indptr_user, indices_user, values_user = convert_structure(data_train_by_user)
    indptr_movie, indices_movie, values_movie = convert_structure(data_train_by_movie)
    indptr_test_user, indices_test_user, values_test_user = convert_structure(data_test_by_user)

    for iteration in range(N):
        update_biases_n_vec_embedding(indptr_user, indices_user, values_user,
                                      user_biases, movie_biases, u, v,
                                      lamda, gamma, tau)

        update_biases_n_vec_embedding(indptr_movie, indices_movie, values_movie,
                                      movie_biases, user_biases, v, u,
                                      lamda, gamma, tau)

        
        r_train, loss_train = cost_function_csr(indptr_user, indices_user, values_user,
                                      user_biases, movie_biases, u, v,
                                      lamda, gamma, tau)

        r_test, loss_test = cost_function_csr(indptr_test_user, indices_test_user, values_test_user,
                                    user_biases, movie_biases, u, v,
                                    lamda, gamma, tau)


        rmse_train.append(r_train)
        costs_train.append(loss_train)
        rmse_test.append(r_test)
        costs_test.append(loss_test)


        if (iteration + 1) % 5 == 0 or iteration == 0:
            duration = time.time() - start_time
            total_duration += duration

            print(f"Iteration {iteration + 1:3d}/{N}\t"
                  f"Train Loss: {loss_train:10.4f}\t"
                  f"Train RMSE: {r_train:6.4f}\t"
                  f"Test Loss: {loss_test:10.4f}\t"
                  f"Test RMSE: {r_test:6.4f}\t"
                  f"Time: {duration:6.2f}s")
            start_time = time.time()

    print(f"Total duration: {total_duration:.2f}s")

    return (user_biases, movie_biases, u, v,
            costs_train, rmse_train, costs_test, rmse_test)



