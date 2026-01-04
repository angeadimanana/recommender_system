import numpy as np
import time

def update_bias(data_train_by_user, user_biases,movie_biases,u, v, k, lamda, gamma,tau):
   for i in range(len(data_train_by_user)):
      bias = 0
      count = 0
      for (n,r) in data_train_by_user[i]:
        bias += lamda * (r - (u[i]@v[n]) - movie_biases[n])
        count += 1
      bias = bias / (lamda * count + gamma)
      user_biases[i] = bias

      s_1 = np.zeros(k)
      s_2 = tau*np.eye(k)
      for (n,r) in data_train_by_user[i]:
        s_1 += v[n]*(r - user_biases[i] - movie_biases[n])
        s_2 += lamda* np.outer(v[n],v[n])
      s_1 = lamda * s_1
      u[i] = np.linalg.solve(s_2,s_1)

   


def cost_function(data_train_by_user, user_biases, movie_biases,u,v,  lamda, gamma,tau):
  m = len(user_biases)
  n = len(movie_biases)

  loss_train = 0
  count = 0
  for x in range(m):
    for (n,r) in data_train_by_user[x]:
        loss_train += (r - u[x]@v[n] - user_biases[x] - movie_biases[n])**2
        count += 1
    
  l_for_u = 0
  l_for_v = 0
  for x in range(m):
    l_for_u += u[x]@u[x]

  for d in range(n):
    l_for_v += v[d]@v[d]

  r_train = np.sqrt(loss_train / count)

  loss_train =  lamda * loss_train /2
  loss_train = loss_train + l_for_u * tau/2
  loss_train = loss_train + l_for_v * tau/2
  loss_train = loss_train + gamma * np.sum(user_biases**2) / 2
  loss_train = loss_train + gamma * np.sum(movie_biases**2) / 2

  return r_train, loss_train

def train(data_train_by_user, data_train_by_movie, data_test_by_user, k, lamda, gamma,tau, N):
    m = len(data_train_by_user)
    n = len(data_train_by_movie)
    user_biases = np.zeros(m)
    movie_biases = np.zeros(n)
    u = np.random.randn(m,k)/np.sqrt(k)
    v = np.random.randn(n,k)/np.sqrt(k) 
   
 
    costs_train = []
    rmse_train_list = []
    costs_test = []
    rmse_test_list = []

    total_duration = 0
    start_time = time.time()
    for tmp in range(N) :
        update_bias(data_train_by_user,  user_biases, movie_biases, u,v, k,lamda, gamma, tau)
        
        update_bias(data_train_by_movie, movie_biases, user_biases, v, u, k, lamda, gamma, tau)

        rmse_train, cout_train = cost_function(data_train_by_user, user_biases, movie_biases, u,v, lamda, gamma, tau)
        rmse_test, cout_test  = cost_function(data_test_by_user, user_biases, movie_biases, u,v, lamda, gamma, tau) 
        
        costs_train.append(cout_train)
        rmse_train_list.append(rmse_train)
        costs_test.append(cout_test)
        rmse_test_list.append(rmse_test)

        if (tmp + 1) % 10 == 0 or tmp == 0:
            duration = time.time() - start_time
            total_duration += duration

            print(f"Iteration {tmp + 1:3d}/{N}\t"
                  f"Train Loss: {cout_train:10.4f}\t"
                  f"Train RMSE: {rmse_train:6.4f}\t"
                  f"Test Loss: {cout_test:10.4f}\t"
                  f"Test RMSE: {rmse_test:6.4f}\t"
                  f"Time: {duration:6.2f}s")
            start_time = time.time()

    print(f"Total duration: {total_duration:.2f}s")


    return user_biases, movie_biases,u,v ,costs_train, rmse_train_list, rmse_test_list, costs_test

