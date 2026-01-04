import numpy as np
from collections import defaultdict

def extract_features(movies_file, movieid_to_idx):
    """
    Extract features (genres) from movies.csv file

    Input:
        movies_file: path to movies.csv
        movieid_to_idx: dictionary mapping movie IDs to indices

    Output:
        feature_to_idx: dict mapping feature name to index
        idx_to_feature: list of feature names
        movie_features: list where movie_features[movie_idx] = list of feature indices
        feature_movies: list where feature_movies[feature_idx] = list of movie indices
    """
    feature_to_idx = {}
    idx_to_feature = []
    movie_features = [[] for _ in range(len(movieid_to_idx))]

    with open(movies_file, 'r', encoding='utf-8') as f:
        next(f)  # Skip header

        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue

            movie_id = parts[0]
            genres_str = parts[-1]  

            if movie_id not in movieid_to_idx:
                continue

            movie_idx = movieid_to_idx[movie_id]

            # Split genres
            if genres_str and genres_str != "(no genres listed)":
                genres = genres_str.split('|')

                for genre in genres:
                    genre = genre.strip()

                    if genre not in feature_to_idx:
                        feature_to_idx[genre] = len(idx_to_feature)
                        idx_to_feature.append(genre)

                    feature_idx = feature_to_idx[genre]
                    movie_features[movie_idx].append(feature_idx)

    num_features = len(idx_to_feature)
    feature_movies = [[] for _ in range(num_features)]

    for movie_idx, features in enumerate(movie_features):
        for feature_idx in features:
            feature_movies[feature_idx].append(movie_idx)

    return feature_to_idx, idx_to_feature, movie_features, feature_movies


def create_feature_matrix(movie_features, num_movies, num_features):
    """
    Create a binary feature matrix F where F[n, i] = 1 if movie n has feature i
    Also returns F_n = number of features per movie

    Input:
        movie_features: list where movie_features[movie_idx] = list of feature indices
        num_movies: total number of movies
        num_features: total number of features

    Output:
        F: binary matrix (num_movies x num_features)
        F_n: array of shape (num_movies,) with count of features per movie
    """
    F = np.zeros((num_movies, num_features), dtype=np.float32)
    F_n = np.zeros(num_movies, dtype=np.int32)

    for movie_idx, features in enumerate(movie_features):
        F_n[movie_idx] = len(features)
        for feature_idx in features:
            F[movie_idx, feature_idx] = 1.0

    return F, F_n

