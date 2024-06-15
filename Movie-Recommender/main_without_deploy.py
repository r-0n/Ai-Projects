import pandas as pd

from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

#STEP 1 preparing, exploring, preprocessing the data after collection
# Loading datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv') 

# Display the first few rows of the data
# print(movies.head())
# print(ratings.head())

# Check for missing values
# print(movies.isnull().sum())
# print(ratings.isnull().sum())

# Merge movies with ratings
data = pd.merge(ratings, movies, on='movieId')

# Display the first few rows of the merged data
print(data.head())


#STEP 2 Choosing an algorithm for the task. In this case a RECOMMENDATION ALGORITHM
# Load data into Surprise format
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Use SVD algorithm
algo = SVD()

# Evaluate the algorithm with cross-validation
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



#STEP 3 TRAINING THE MODEL 
# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Train the algorithm on the trainset
algo.fit(trainset)

# Test the algorithm on the testset
predictions = algo.test(testset)

# Calculate RMSE
from surprise import accuracy
accuracy.rmse(predictions)



#STEP 4 GENERATING OUTPUT
# Function to get top-N recommendations for a user
def get_top_n_recommendations(user_id, n=15):
    # Get a list of all movieIds
    movie_ids = movies['movieId'].tolist()
    
    # Predict ratings for all movies not seen by the user
    unseen_movies = [movie_id for movie_id in movie_ids if movie_id not in ratings[ratings['userId'] == user_id]['movieId'].tolist()]
    predictions = [algo.predict(user_id, movie_id) for movie_id in unseen_movies]
    
    # Sort predictions by estimated rating
    top_n_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    
    # Get movie titles
    top_n_movie_ids = [pred.iid for pred in top_n_predictions]
    top_n_movies = movies[movies['movieId'].isin(top_n_movie_ids)]
    
    return top_n_movies

# Get top 10 recommendations for user with userId=1
top_n_recommendations = get_top_n_recommendations(user_id=1, n=10)
print(top_n_recommendations)



