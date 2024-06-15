from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

def build_model(data):
    # Load data into Surprise format
    reader = Reader(rating_scale=(0.5, 5.0))
    dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
    
    # Split the data into training and testing sets
    trainset, testset = train_test_split(dataset, test_size=0.25)
    
    # Use the SVD algorithm
    algo = SVD()
    algo.fit(trainset)
    
    # Evaluate the model
    predictions = algo.test(testset)
    accuracy.rmse(predictions)
    
    return algo

def get_top_n_recommendations(algo, user_id, movies, ratings, n=10):
    # Get a list of all movie IDs
    movie_ids = movies['movieId'].tolist()
    
    # Get movies that the user has not seen yet
    unseen_movies = [movie_id for movie_id in movie_ids if movie_id not in ratings[ratings['userId'] == user_id]['movieId'].tolist()]
    
    # Predict ratings for unseen movies
    predictions = [algo.predict(user_id, movie_id) for movie_id in unseen_movies]
    
    # Sort predictions by estimated rating in descending order
    top_n_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    
    # Get the movie IDs of the top-N recommendations
    top_n_movie_ids = [pred.iid for pred in top_n_predictions]
    
    # Get the movie details of the top-N recommendations
    top_n_movies = movies[movies['movieId'].isin(top_n_movie_ids)]
    
    return top_n_movies
