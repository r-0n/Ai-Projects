import pandas as pd
import numpy as np
import yfinance as yf

# Download historical stock price data for a specific ticker
ticker = 'AAPL'  # Example ticker for Apple Inc.
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
data.to_csv('Stock-Price-Predictor/stock_prices.csv') 

def load_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Handle missing values
    data = data.dropna()
    
    # Feature engineering: create additional features if necessary
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['Price_Change'] = data['Close'].diff()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    # Drop rows with NaN values created by rolling means
    data = data.dropna()
    
    return data

def create_features(data):
    # Use 'Close' as the target variable
    X = data[['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_50']]
    y = data['Close']
    
    return X, y
