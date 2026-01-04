
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

from src.data_struct.split_train_test import split_train_test, split_train_test_temporal_ratio
from src.training.features_function import extract_features, create_feature_matrix

def load_and_split_data(ratings_file='data/ml-32m/ratings.csv',
                        movies_file='data/ml-32m/movies.csv',
                        test_ratio=0.2):
    """
    Load data and split into train/test (80/20)
    """
    movies_df = pd.read_csv(movies_file)
    # Split data
    (userid_to_idx, idx_to_userid, data_train_by_user, data_test_by_user,
     movieid_to_idx, idx_to_movieid, data_train_by_movie, data_test_by_movie) = \
        split_train_test(ratings_file, 1.0 - test_ratio)

    # Extract features
    feature_to_idx, idx_to_feature, movie_features, feature_movies = \
        extract_features(movies_file, movieid_to_idx)

    num_movies = len(movieid_to_idx)
    num_features = len(idx_to_feature)
    F, F_n = create_feature_matrix(movie_features, num_movies, num_features)

    print(f"Data loaded:")
    print(f"  Users: {len(userid_to_idx)}")
    print(f"  Movies: {num_movies}")
    print(f"  Features: {num_features}")
    print(f"  Train: {sum(len(r) for r in data_train_by_user)} ratings")
    print(f"  Test: {sum(len(r) for r in data_test_by_user)} ratings\n")

    return {
        'userid_to_idx': userid_to_idx,
        'idx_to_userid': idx_to_userid,
        'movieid_to_idx': movieid_to_idx,
        'idx_to_movieid': idx_to_movieid,
        'data_train_by_user': data_train_by_user,
        'data_train_by_movie': data_train_by_movie,
        'data_test_by_user': data_test_by_user,
        'data_test_by_movie': data_test_by_movie,
        'movies_df': movies_df,
        'F': F,
        'F_n': F_n,
        'feature_to_idx': feature_to_idx,
        'idx_to_feature': idx_to_feature,
        'movie_features': movie_features
    }

def main():
    # Load data 
    data = load_and_split_data()
    N = 50  # Number of iterations

    from src.visualization.plot_loss_rmse import plot

    # train for bias only
    from src.training.train_bias_only import train_numba as train_bias

    lamda_bias = 0.01
    gamma_bias = 0.01

    results_bias = train_bias(
        data['data_train_by_user'],
        data['data_train_by_movie'],
        data['data_test_by_user'],
        lamda_bias, gamma_bias, N
    )
    plot(results_bias['costs_train'], results_bias['costs_test'],
        results_bias['rmse_train'], results_bias['rmse_test'])

    k = 10 
    #train for embedding vector and bias
    from src.training.train_parallelize import train as train_bias_embedding
    lamda_emb = 0.1
    gamma_emb = 0.05
    tau_emb = 1.0

    results_embedding = train_bias_embedding(
        data['data_train_by_user'],
        data['data_train_by_movie'],
        data['data_test_by_user'],
        k, lamda_emb, gamma_emb, tau_emb, N
    )
    plot(results_embedding['costs_train'], results_embedding['costs_test'],
        results_embedding['rmse_train'], results_embedding['rmse_test'])

    # train with adding features
    from src.training.train_with_features import train_with_features 
    lamda_feat = 0.1
    gamma_feat = 0.01
    tau_feat = 1.0

    results_features = train_with_features(
        data['data_train_by_user'],
        data['data_train_by_movie'],
        data['data_test_by_user'],
        data['F'], data['F_n'],
        k, lamda_feat, gamma_feat, tau_feat, N
    )

    plot(results_features['costs_train'], results_features['costs_test'],
        results_features['rmse_train'], results_features['rmse_test'])

    from src.visualization.plot_compare import plot_compare_costs, plot_compare_rmse
    plot_compare_costs(results_bias, results_embedding, results_features)
    plot_compare_rmse(results_bias, results_embedding, results_features)

    # ------------------------------------------------------------------------------------------------

    selected_movies = {
        'Action/Adventure': ['Lord of the Rings, The (1978)', 'Lord of the Rings: The Fellowship of',
                             'Lord of the Rings: The Two Towers', 'Lord of the Rings: The Return of the King'],
        'Family/Animation': ['Toy Story', 'Toy Story 2', 'Toy Story 3'],
        'Comedy' : ['Home Alone', 'Home Alone 2', 'Home Alone 3'],
        'Horror': [ 'Halloween', 'Friday the 13th', 'Alien']
    }

    def find_movie_idx(title_part, movies_df, movieid_to_idx):
        """Find movie index by title match"""
        matches = movies_df[movies_df['title'].str.contains(title_part, case=False, na=False, regex=False)]
        if len(matches) > 0:
            movie_id = str(matches.iloc[0]['movieId'])
            if movie_id in movieid_to_idx:
                return movieid_to_idx[movie_id], matches.iloc[0]['title']
        return None, None


    # find movie according to the selected movie
    movie_indices = {}
    movie_titles_full = {}
    for genre, titles in selected_movies.items():
        for title in titles:
            idx, full_title = find_movie_idx(title, data['movies_df'], data['movieid_to_idx'])
            if idx is not None:
                movie_indices.setdefault(genre, []).append(idx)
                movie_titles_full.setdefault(genre, []).append(full_title)

    # t-sne to 2D for selected movies
    v_emb = results_features['v']
    selected_indices = [idx for indices in movie_indices.values() for idx in indices]
    v_selected = v_emb[selected_indices]
    tsne = TSNE(n_components=2, random_state=100, perplexity=10, max_iter=1000)
    v_selected_2d = tsne.fit_transform(v_selected)

    tsne_coords = dict(zip(selected_indices, v_selected_2d))

    plt.figure(figsize=(6, 4))
    plt.rcParams.update({'font.size': 9})

    colors = {
        'Action/Adventure':'red', 
        'Comedy': 'blue',           
        'Family/Animation': 'green', 
        'Horror': 'purple'            
    }


    for genre, indices in movie_indices.items():
        for i, idx in enumerate(indices):
            x, y = tsne_coords[idx]
            plt.scatter(x, y, c=colors[genre], s=60, alpha=0.8,
                        edgecolors='black', linewidth=0.8)
            plt.annotate(movie_titles_full[genre][i][:15] + '...',  
                        (x, y), fontsize=7, fontweight='bold', alpha=0.9,
                        xytext=(4, 4), textcoords='offset points')

    plt.xlabel('ax1', fontsize=10, fontweight='bold')
    plt.ylabel('ax2', fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=colors[g], markersize=7, label=g)
                    for g in colors]
    plt.legend(handles=legend_elements, fontsize=8, loc='best', frameon=True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------------------------------

    # features embedding in 2D


    feature_vectors = results_features['f_vectors']
    feature_name = [f_name for f_name in data['feature_to_idx'].keys()]

    tsne = TSNE(n_components=2, random_state=42, perplexity=4, max_iter=1000)
    features_2d = tsne.fit_transform(feature_vectors)



    plt.figure(figsize=(6, 5))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], s=100, alpha=0.6)

    for i, name in enumerate(feature_name):
        plt.annotate(name, (features_2d[i, 0], features_2d[i, 1]),
                    fontsize=8, fontweight='bold',ha='center')

    plt.xlabel('ax1', fontsize=10)
    plt.ylabel('ax2', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------------------------------------------

    # polarizing movies
    from src.prediction.find_polarizing import find_polarizing_movies


    polarizing_features = find_polarizing_movies(
        data['movies_df'],
        v_emb=results_features['v'],
        b_movie=results_features['movie_biases'],
        data_by_movie=data['data_train_by_movie'],
        idx_to_movieid=data['idx_to_movieid'],
        min_ratings=500,
        top_k=20
    )

    print("\nTop 20 Most Polarizing Movies:")

    print("-"*80)
    print(f"{'Rank':<5} {'Title':<50} {'||v||':<8} {'Bias':<8} {'Number of Ratings':<10}")
    print("-"*80)

    for i, row in polarizing_features.iterrows():
        print(f"{i+1:<5} {row['title'][:48]:<50} {row['v_norm']:<8.3f} {row['bias']:<8.3f} {row['num_ratings']:<10.0f}")

    # -------------------------------------------------------------------------------------------------------------------
    # recommendations
    from src.prediction.recommendation import get_user_recommendations_embedding

    # example of recommendation, 

    # a user rates 5/5 the movie Lord of The Rings: The (1978)
    dummy_data = [(1819, 5.0)]
    movie_idx_rated = [x[0] for x in dummy_data]

    recomm = get_user_recommendations_embedding(dummy_data, results_features['movie_biases'],
                                    results_features['v'], movie_idx_rated, data['idx_to_movieid'],
                                    data['movies_df'], k, lamda_feat, gamma_feat,
                                    tau_feat, top_k=10, bias_weight=0.05)
    movie_recommendation = pd.DataFrame(recomm)

    print(f"The recommended movies are: ")
    print(movie_recommendation)