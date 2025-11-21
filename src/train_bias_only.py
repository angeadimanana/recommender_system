import numpy as np

def update_bias(data_train_by_user, lamda, gamma, movie_biases):
    m = len(data_train_by_user)
    
    biases = np.zeros(m)

    for i in range(m):
        bias = 0
        count = 0
        for (n,r) in data_train_by_user[i]:
            bias += lamda * (r - movie_biases[n])
            count += 1
        bias = bias / (lamda * count + gamma)
        biases[i] = bias
    
    return biases


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

def train(data_train_by_user, data_train_by_movie, data_test_by_user, data_test_by_movie, lamda, gamma, N):
    
    user_biases = np.zeros(len(data_train_by_user))
    movie_biases = np.zeros(len(data_train_by_movie))
    
    costs_train = []
    rmse_train_list = []
    costs_test = []
    rmse_test_list = []

    for tmp in range(N) :
        update_bias(data_train_by_user, lamda,gamma, movie_biases)
        update_bias(data_train_by_movie, lamda, gamma, user_biases)

        rmse_train, cout_train = cost_function(data_test_by_user, user_biases, movie_biases, lamda, gamma)
        rmse_test, cout_test  = cost_function(data_test_by_user, user_biases, movie_biases, lamda, gamma) 
        
        costs_train.append(cout_train)
        rmse_train_list.append(rmse_train)
        costs_test.append(cout_test)
        rmse_test_list.append(rmse_test)

    return user_biases, movie_biases, costs_train, rmse_train_list, rmse_test_list, costs_test


