import numpy as np
import pandas as pd
from src.data_struct.data_structure import convert_structure
from src.training.train_parallelize import update_biases_n_vec_embedding
from src.training.train_bias_only import update_bias_numba 
from src.training.train_with_features import update_user_biases_and_vectors_with_features

# for bias only
def recommend_bias_only(user_bias, movie_biases,
                       idx_to_movieid, movie_idx_rated, movies_df, top_k=10):
    """
    Make recommendations using bias-only model

    Args:
        user_biases: User bias vector
        movie_biases: Movie bias vector
        user_idx: Index of the user
        idx_to_movieid: Mapping from movie index to movie ID
        movies_df: DataFrame with movie information
        top_k: Number of recommendations

    Returns:
        List of recommendations
    """
    scores = user_bias + movie_biases
    #scores = user_biases[user_idx] + movie_biases

    # Rank by descending score
    top_indices = np.argsort(scores)[::-1]

    recommendations = []
    for movie_idx in top_indices:
        if movie_idx in movie_idx_rated:
          continue
        if len(recommendations) >= top_k:
            break

        movie_id = idx_to_movieid[movie_idx]
        score = scores[movie_idx]

        # Get movie details
        title = f"Movie {movie_id}"
        genres = ""

        if movies_df is not None:
            movie_row = movies_df[movies_df['movieId'] == int(movie_id)]
            if not movie_row.empty:
                title = movie_row.iloc[0]['title']
                genres = movie_row.iloc[0].get('genres', '')

        recommendations.append({
            'rank': len(recommendations) + 1,
            'movie_idx': movie_idx,
            'movie_id': movie_id,
            'title': title,
            'genres': genres,
            'score': score
        })

    return recommendations

def get_user_recommendations(data_user, movie_biases,
                             idx_to_movieid, movie_idx_rated ,movies_df,
                             lamda, gamma, top_k=10):


    data_by_user = [data_user]
    indptr, indices, values = convert_structure(data_by_user)


    user_bias = np.zeros(1, dtype=np.float64)


    for _ in range(10):
      update_bias_numba(indptr, indices, values,
                        user_bias, movie_biases,
                        lamda, gamma)

    recommendations = recommend_bias_only(user_bias, movie_biases,
                       idx_to_movieid, movie_idx_rated,movies_df, top_k)

    return recommendations

# case with embedding
def make_recommendations(user_id,  movie_biases, u, v,
                         movie_idx_rated, idx_to_movieid, data_by_movie,movies_df,
                        top_k, bias_weight=0.05):


    scores = np.dot(v, u[user_id]) + bias_weight * movie_biases

    # rank the result to descending order
    top_recommandation = np.argsort(scores)[::-1]

    recommendations = []
    for movie_idx in top_recommandation:
        if movie_idx in movie_idx_rated:
          continue
        
        if len(data_by_movie[movie_idx]) < 100:
            continue
        if len(recommendations) >= top_k:
            break

        movie_id = idx_to_movieid[movie_idx]
        score = scores[movie_idx]

        # movie detail
        title = f"Movie {movie_id}"
        genres = ""

        if movies_df is not None:
            movie_row = movies_df[movies_df['movieId'] == int(movie_id)]
            if not movie_row.empty:
                title = movie_row.iloc[0]['title']
                genres = movie_row.iloc[0].get('genres', '')

        recommendations.append({
            'movie_idx': movie_idx,
            'movie_id': movie_id,
            'title': title,
            'genres': genres,
            'score': score
        })

    return recommendations



def get_user_recommendations_embedding(data_user, movie_biases, v,
                                  movie_idx_rated, idx_to_movieid, movies_df,
                                  k, lamda, gamma, tau,
                                  top_k=20, bias_weight=0.05, model='with_features'):


    data_by_user = [data_user]
    indptr, indices, values = convert_structure(data_by_user)


    user_bias = np.zeros(1, dtype=np.float64)
    u = np.random.randn(1, k).astype(np.float64) / np.sqrt(k)


    for _ in range(10):
        if model == 'with_features':
            update_user_biases_and_vectors_with_features(
            indptr, indices, values,
            user_bias, movie_biases, u, v,
            lamda, gamma, tau
        )
            
        else:
             update_biases_n_vec_embedding(
                indptr, indices, values,
                user_bias, movie_biases, u, v,
                lamda, gamma, tau
            )
    recommendations = make_recommendations(0, movie_biases, u, v,
        movie_idx_rated, idx_to_movieid, movies_df,top_k, bias_weight
    )

    return recommendations


