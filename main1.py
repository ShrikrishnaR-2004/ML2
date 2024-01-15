import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

# Load the dataset
ratings = pd.read_csv('ratings.csv')

# Load the movie titles
movie_titles = pd.read_csv('Movie_Id_Titles.csv')

# Convert movie titles to movie IDs
ratings['movieId'] = ratings['movieId'].map(movie_titles.set_index('movieId')['title'])

# Create a user-item matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)


# User-based collaborative filtering
def get_recommendations(user_id, n_recommendations):
    # Calculate the similarity between users
    user_similarity = 1 - pairwise_distances(user_item_matrix.T, metric='cosine')

    # Find the k-nearest neighbors
    knn = NearestNeighbors(n_neighbors=n_recommendations + 1, metric='precomputed').fit(user_similarity)
    distances, indices = knn.kneighbors(user_id)

    # Get the recommended movies
    recommended_movies = []
    for i in range(1, len(indices)):
        similarity = 1 - distances[0][i]
        recommended_movies.append((user_item_matrix.columns[indices[0][i]], similarity))

    # Sort by similarity and take the top n recommendations
    recommended_movies = sorted(recommended_movies, key=lambda x: x[1], reverse=True)[:n_recommendations]

    # Get the movie titles
    recommended_movies = [(movie_titles[movie_titles['title'] == movie]['movieId'].values[0], similarity) for
                          movie, similarity in recommended_movies]

    return recommended_movies


# Example usage
user_id = 5
n_recommendations = 10
recommendations = get_recommendations(user_id, n_recommendations)
print(f'Recommendations for user {user_id}:')
for movie, similarity in recommendations:
    print(f'{movie}: {similarity:.2f}')