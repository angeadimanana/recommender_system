import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_data_structure(filename):
  """
  It builds structure for the data in list[list],
  one indexed by the userId, and another one by the movieId
  Input:
    filename: path of the csv file
    split_ratio: the proportionality of the two split, in respect
                to the train

  Output:
    list of index of user,
    list of user,
    list of the data_train by userId,
    list of the data_test by userId,

    list of index of movie,
    list of movie,
    list of the data_train by movieId,
    list of the data_test by movieId,
  """

  userid_to_idx = {} # mapping the userid to index
  idx_to_userid = [] # mapping index to userid
  data_by_user =[] # store the moviename and its rate according the index of userid
  

  movieid_to_idx ={} # mapping the moveid to index
  idx_to_movieid = [] # mapping index to movieid
  data_by_movie = [] # store the userid and rate of correspind movie
  data_by_movie = []

  with open(filename, 'r', encoding="utf-8") as rate:
    next(rate)  # Skip header

    for line in rate:
        v = line.strip().split(',')
        user_id, movie_id, rating = v[0], v[1], float(v[2])

        if user_id not in userid_to_idx:
            userid_to_idx[user_id] = len(idx_to_userid)
            idx_to_userid.append(user_id)
            data_by_user.append([])
         

        if movie_id not in movieid_to_idx:
            movieid_to_idx[movie_id] = len(idx_to_movieid)
            idx_to_movieid.append(movie_id)
            data_by_movie.append([])
           

        user_idx = userid_to_idx[user_id]
        movie_idx = movieid_to_idx[movie_id]

        data_by_user[user_idx].append((movie_idx, rating))
        data_by_movie[movie_idx].append((user_idx, rating))
      
  return (userid_to_idx, idx_to_userid, movieid_to_idx, 
          idx_to_movieid,data_by_user, data_by_movie)


