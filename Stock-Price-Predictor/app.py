import streamlit as st
import pandas as pd
from data_processing import load_data, preprocess_data, create_features
from model import train_model
from prediction import predict

# Load and preprocess data
data = load_data('Stock-Price-Predictor/stock_prices.csv')
data = preprocess_data(data)
X, y = create_features(data)

# Train model
model, scaler, mse, mae = train_model(X, y)

# Streamlit app
st.title('Stock Price Prediction')

st.write(f'Mean Squared Error: {mse}')
st.write(f'Mean Absolute Error: {mae}')

st.subheader('Predict Stock Price')
open_price = st.number_input('Open Price:', min_value=0.0, format="%.2f")
high_price = st.number_input('High Price:', min_value=0.0, format="%.2f")
low_price = st.number_input('Low Price:', min_value=0.0, format="%.2f")
volume = st.number_input('Volume:', min_value=0, format="%d")
ma_10 = st.number_input('10-Day Moving Average:', min_value=0.0, format="%.2f")
ma_50 = st.number_input('50-Day Moving Average:', min_value=0.0, format="%.2f")

if st.button('Predict'):
    input_data = [open_price, high_price, low_price, volume, ma_10, ma_50]
    prediction = predict(model, scaler, input_data)
    st.write(f'Predicted Close Price: {prediction:.2f}')
