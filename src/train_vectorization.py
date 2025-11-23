import numpy as np
import time

def update_biases_n_vec_embedding(data_train, u_biases, m_biases, u, v, k, lamda, gamma, tau):
    for i in range(len(u_biases)):
        if len(data_train[i]) == 0:
            continue

        # transform the list to numpy
        data_array = np.array(data_train[i])
        indices = data_array[:, 0].astype(int)  # user/movie index
        rating = data_array[:, 1]  # Ratings

        #vectorization of update of bias
        V_subset = v[indices]
        prod_scal = V_subset @ u[i]
        res = rating - prod_scal - m_biases[indices]
        bias = lamda * np.sum(res)
        u_biases[i] = bias / (lamda * len(data_train[i]) + gamma)

        #vectorization for update of embedding vector
        val = rating - u_biases[i] - m_biases[indices]
        s_1 = lamda * (V_subset.T @ val)
        s_2 = tau * np.eye(k) + lamda * (V_subset.T @ V_subset)
        u[i] = np.linalg.solve(s_2, s_1)

def regularization_vect(u, tau):
    return tau * np.sum(u**2) / 2


def regularization_biases(bias, gamma):
    return gamma * np.sum(bias**2) / 2



def cost_function(data_train, u_biases, m_biases, u, v, lamda, gamma, tau):
    
    squared_error = 0
    count = 0

    for x in range(len(data_train)):
        if len(data_train[x]) == 0:
            continue

        data_array = np.array(data_train[x])
        indices = data_array[:, 0].astype(int)
        rating = data_array[:, 1]

        # Predict
        V_subset = v[indices]
        predictions = (V_subset @ u[x]) + u_biases[x] + m_biases[indices]

        squared_error += np.sum((rating - predictions)**2)
        count += len(rating)

    
    if count != 0 :
         rmse = np.sqrt(squared_error / count)
    else:
         rmse = 0

    loss = lamda * squared_error / 2
    loss += regularization_biases(u_biases, gamma)
    loss += regularization_biases(m_biases, gamma)
    loss += regularization_vect(u, tau)
    loss += regularization_vect(v, tau)

    return rmse, loss


def train(data_train_by_user, data_train_by_movie,data_test_by_user, 
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

    for iteration in range(N):

        update_biases_n_vec_embedding(data_train_by_user, user_biases, movie_biases,
                                       u, v, k, lamda, gamma, tau)
        update_biases_n_vec_embedding(data_train_by_movie, movie_biases, user_biases,
                                       v, u, k, lamda, gamma, tau)


        r_train, loss_train = cost_function(data_train_by_user, user_biases, movie_biases,
                                             u, v, lamda, gamma, tau)
        r_test, loss_test = cost_function(data_test_by_user, user_biases, movie_biases,
                                           u, v, lamda, gamma, tau)

        rmse_train.append(r_train)
        costs_train.append(loss_train)
        rmse_test.append(r_test)
        costs_test.append(loss_test)


        if (iteration + 1) % 10 == 0 or iteration == 0:
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




