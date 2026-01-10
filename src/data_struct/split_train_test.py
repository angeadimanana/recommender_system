import numpy as np

def split_train_test(filename, split_ratio,seed=42):
    """
    It builds structure for the data and split it in two sets:
    train and test with ratio (e.g 0.8 --> Train: 80%; Test: 20%)

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
    np.random.seed(seed)
    userid_to_idx = {} # mapping the userid to index
    idx_to_userid = [] # mapping index to userid
    data_train_by_user =[] # store the moviename and its rate according the index of userid
    data_test_by_user = []

    movieid_to_idx ={} # mapping the moveid to index
    idx_to_movieid = [] # mapping index to movieid
    data_train_by_movie = [] # store the userid and rate of correspind movie
    data_test_by_movie = []

    line_count = 0

    with open(filename, 'r', encoding="utf-8") as rate:
        next(rate)  # Skip header

        for line in rate:
            v = line.strip().split(',')
            user_id, movie_id, rating = v[0], v[1], float(v[2])

            if user_id not in userid_to_idx:
                userid_to_idx[user_id] = len(idx_to_userid)
                idx_to_userid.append(user_id)
                data_train_by_user.append([])
                data_test_by_user.append([])

            if movie_id not in movieid_to_idx:
                movieid_to_idx[movie_id] = len(idx_to_movieid)
                idx_to_movieid.append(movie_id)
                data_train_by_movie.append([])
                data_test_by_movie.append([])

            user_idx = userid_to_idx[user_id]
            movie_idx = movieid_to_idx[movie_id]

            u = np.random.random()

            if u < split_ratio:
                data_train_by_user[user_idx].append((movie_idx, rating))
                data_train_by_movie[movie_idx].append((user_idx, rating))
            else:
                data_test_by_user[user_idx].append((movie_idx, rating))
                data_test_by_movie[movie_idx].append((user_idx, rating))
            
            line_count += 1
            
            magnitude = max(10 ** (len(str(line_count)) - 1), 100000) 
            if line_count % magnitude == 0:
                print(f"Loaded {line_count} lines...")
            
    print(f"Loading complete: {line_count} lines processed")    
    
    return (userid_to_idx, idx_to_userid, data_train_by_user, data_test_by_user,movieid_to_idx, 
            idx_to_movieid, data_train_by_movie, data_test_by_movie)




def split_train_test_temporal_ratio(filename, test_ratio=0.2):
    """
      Splits data temporally by taking the most recent ratings of each user for test.
      For each user, sort the ratings by timestamp and take the last ratings according 
      to the ratio  for the test set. 

      Input:
          filename: path of the csv file
          test_ratio: proportion of most recent ratings to use for test (e.g., 0.2 = 20%)

      Output:
          userid_to_idx: mapping from user_id to index
          idx_to_userid: mapping from index to user_id
          data_train_by_user: list of training ratings indexed by user
          data_test_by_user: list of test ratings indexed by user
          movieid_to_idx: mapping from movie_id to index
          idx_to_movieid: mapping from index to movie_id
          data_train_by_movie: list of training ratings indexed by movie
          data_test_by_movie: list of test ratings indexed by movie
    """
    userid_to_idx = {}
    idx_to_userid = []
    movieid_to_idx = {}
    idx_to_movieid = []
    
    user_ratings = {}
    
    line_count = 0

    with open(filename, 'r', encoding="utf-8") as rate:
        next(rate)
        for line in rate:
            v = line.strip().split(',')
            user_id, movie_id, rating, timestamp = v[0], v[1], float(v[2]), int(v[3])
            
            if user_id not in userid_to_idx:
                userid_to_idx[user_id] = len(idx_to_userid)
                idx_to_userid.append(user_id)
            
            if movie_id not in movieid_to_idx:
                movieid_to_idx[movie_id] = len(idx_to_movieid)
                idx_to_movieid.append(movie_id)
            
            if user_id not in user_ratings:
                user_ratings[user_id] = []
            user_ratings[user_id].append((movie_id, rating, timestamp))
    
    num_users = len(idx_to_userid)
    num_movies = len(idx_to_movieid)
    data_train_by_user = [[] for tmp in range(num_users)]
    data_test_by_user = [[] for tmp in range(num_users)]
    data_train_by_movie = [[] for tmp in range(num_movies)]
    data_test_by_movie = [[] for tmp in range(num_movies)]
    
    for user_id, ratings in user_ratings.items():
        user_idx = userid_to_idx[user_id]
        
        # Sort by timestamp
        ratings.sort(key=lambda x: x[2])
        
        # Calculate split point based on ratio
        num_ratings = len(ratings)
        num_test = max(1, int(num_ratings * test_ratio))  # At least 1 for test
        separation = num_ratings - num_test
        
        train_ratings = ratings[:separation]
        test_ratings = ratings[separation:]
        
        for movie_id, rating, _ in train_ratings:
            movie_idx = movieid_to_idx[movie_id]
            data_train_by_user[user_idx].append((movie_idx, rating))
            data_train_by_movie[movie_idx].append((user_idx, rating))
        
        for movie_id, rating, _ in test_ratings:
            movie_idx = movieid_to_idx[movie_id]
            data_test_by_user[user_idx].append((movie_idx, rating))
            data_test_by_movie[movie_idx].append((user_idx, rating))
    

    
            line_count += 1
            
            magnitude = max(10 ** (len(str(line_count)) - 1), 100000) 
            if line_count % magnitude == 0:
                print(f"Loaded {line_count} lines...")
            
    print(f"Loading complete: {line_count} lines processed") 

    return (userid_to_idx, idx_to_userid, data_train_by_user, data_test_by_user,
            movieid_to_idx, idx_to_movieid, data_train_by_movie, data_test_by_movie)


