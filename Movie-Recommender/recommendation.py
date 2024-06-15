from data_processing import load_data, preprocess_data
from model import build_model, get_top_n_recommendations

def get_recommendations(user_id, n):
    # Load and preprocess data
    data, movies, ratings = load_data('Movie-Recommender/ml-latest-small/movies.csv', 'Movie-Recommender/ml-latest-small/ratings.csv')
    preprocessed_data = preprocess_data(data)
    
    # Build and train model
    algo = build_model(preprocessed_data)
    
    # Get recommendations
    recommendations = get_top_n_recommendations(algo, user_id, movies, ratings, n)
    
    return recommendations
