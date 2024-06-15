import streamlit as st

from recommendation import get_recommendations 

st.title('Movie Recommendation System')

# Get user input
user_id = st.number_input('Enter User ID:', min_value=1, value=1, step=1)
n = st.number_input('Enter number of recommendations:', min_value=1, value=10, step=1)

if st.button('Get Recommendations'):
    recommendations = get_recommendations(user_id, n)
    st.write('Top {} movie recommendations for User ID {}:'.format(n, user_id))
    st.write(recommendations)
