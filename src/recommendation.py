import numpy as np
import pandas as pd
from train_parallelize import *


def make_recommendations(user_id,  movie_biases, u, v,
                         idx_to_movieid, movies_df,
                        top_k, bias_weight=0.05):


    scores = np.dot(v, u[user_id]) + bias_weight * movie_biases

    # rank the result to descending order
    top_recommandation = np.argsort(scores)[::-1]

    recommendations = []
    for movie_idx in top_recommandation:
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



def get_user_recommendations(data_user, movie_biases, v,
                                  idx_to_movieid, movies_df,
                                  k, lamda, gamma, tau,
                                  top_k=20, bias_weight=0.05):


    data_by_user = [data_user]
    indptr, indices, values = convert_structure(data_by_user)


    user_bias = np.zeros(1, dtype=np.float64)
    u = np.random.randn(1, k).astype(np.float64) / np.sqrt(k)


    for _ in range(100):
        update_biases_n_vec_embedding(
            indptr, indices, values,
            user_bias, movie_biases, u, v,
            lamda, gamma, tau
        )

    recommendations = make_recommendations(0, movie_biases, u, v,
        idx_to_movieid, movies_df,top_k, bias_weight
    )

    return recommendations, user_bias[0], u[0]


