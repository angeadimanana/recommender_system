import pandas as pd
import numpy as np

def find_polarizing_movies(movies_df,  v_emb, b_movie, 
                                    data_by_movie, idx_to_movieid, 
                                   min_ratings=500, top_k=20):
    """
    Find polarizing movies using:
    high standard deviation in the ratings and high L2 norm
    Input: 
        movies_df : dataframe of the movies_file (movieId, title, genre)
        v_emb : matrix of embedding vectors of the movie
        b_movie : array, bias of the movie
        data_by_movie: structure of the ratings and users which rated the movie
        idx_to_movieid: structure linking the index in data_by_movie to movieId
        min_ratings: integer to filter movie having at least those ratings
        top_k : the top_k polarizing
    
    Output: 
        dataframe the k polarizing movies
    """
    
    results = []
    for movie_idx in range(len(v_emb)):
        ratings_list = data_by_movie[movie_idx]
        
        if len(ratings_list) < min_ratings:
            continue
            
        ratings = [r for _, r in ratings_list]
        mean_rating = np.mean(ratings)
        std_rating = np.std(ratings)
        
        movie_id = idx_to_movieid[movie_idx]
        movie_title = movies_df[movies_df['movieId'] == int(movie_id)]['title'].values
        
        if len(movie_title) > 0:
            v_norm = np.linalg.norm(v_emb[movie_idx])
            bias = b_movie[movie_idx]
            
            results.append({
                'movie_idx': movie_idx,
                'title': movie_title[0],
                'v_norm': v_norm,
                'bias': bias,
                'abs_bias': abs(bias),
                'mean': mean_rating,
                'std': std_rating,
                'num_ratings': len(ratings)
            })
    
    df = pd.DataFrame(results)
    
    polarizing = df[(df['std'] > 1.15)]
    polarizing = polarizing.sort_values('v_norm', ascending=False)
    
    return polarizing.head(top_k)