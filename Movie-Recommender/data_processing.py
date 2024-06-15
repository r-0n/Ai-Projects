import pandas as pd

def load_data(movies_path, ratings_path):
    # Load datasets
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    # Merge the datasets
    data = pd.merge(ratings, movies, on='movieId')
    
    return data, movies, ratings

def preprocess_data(data):
    # Perform any additional preprocessing steps if necessary
    # For now, we simply return the data
    return data
