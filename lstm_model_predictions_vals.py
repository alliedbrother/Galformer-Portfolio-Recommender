# %%
import yfinance as yf
import pandas as pd
import datetime
from fredapi import Fred
import numpy as np
import pandas_ta as ta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'


# %%
def prepare_inference_data(company_df, sequence_length=60):
    """
    Prepare input data for inference .
    Args:
        company_df (DataFrame): The DataFrame for a specific company.
        sequence_length (int): The number of past days to consider as input.

    Returns:
        numpy array: The input data ready for prediction.
    """
    # Ensure data is sorted by date
    company_df = company_df.sort_index()

    # Select relevant input features (exclude targets)
    input_features = company_df.filter(regex="^(?!.*target).*").values

    # Take the last `sequence_length` days as input for prediction
    if len(input_features) >= sequence_length:
        input_sequence = input_features[-sequence_length:]
        return np.expand_dims(input_sequence, axis=0)  # Add batch dimension
    else:
        raise ValueError("Insufficient data for inference (less than sequence length).")

# %%
def get_predictions_for_all_companies(all_dfs, model, ticker, sequence_length=60):
    """
    Get predictions for all companies using the trained model.
    Args:
        all_dfs (dict): Dictionary of company DataFrames.
        model: Trained LSTM model.
        sequence_length (int): Number of past days to consider as input.

    Returns:
        dict: Predictions for each company.
    """
    predictions = {}

    all_dfs = {k: v for k, v in all_dfs.items() if k == "df_"+ticker}
    
    for company, df in all_dfs.items():
        try:
            # Prepare data for inference
            input_data = prepare_inference_data(df, sequence_length=sequence_length)
            
            # Make predictions
            pred = model.predict(input_data)
            
            # Store predictions
            predictions[company] = pred.flatten()  # Flatten the array for readability
        
        except ValueError as e:
            print(f"Skipping {company}: {e}")

    return predictions

# %%
def data_loading():
    start_date = (datetime.datetime.today() - datetime.timedelta(days=100)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    # Example list of S&P 500 tickers (full list can be obtained elsewhere)
    tickers = ["AAPL", "NVDA", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "LLY", "TSLA", 
                    "WMT", "JPM", "V", "XOM", "UNH", "ORCL", "MA", "HD", "PG", "COST", "JNJ", 
                    "NFLX", "ABBV", "BAC", "KO", "CRM", "CVX", "MRK", "TMUS", "AMD", "PEP", 
                    "ACN", "LIN", "TMO", "MCD", "CSCO", "ADBE", "WFC", "IBM", "GE", "ABT", 
                    "DHR", "AXP", "MS", "CAT", "NOW", "QCOM", "PM", "ISRG", "VZ"]

    # Download data for all tickers at once
    stock_price_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

    stock_data = stock_price_data.copy()

    # Flatten MultiIndex columns in stock_data
    stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

    # Convert index to DatetimeIndex if not already
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data.index)

    # Sort by date
    stock_data.sort_index(inplace=True)

    with open('updated_all_dfs.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    
    inference_data = {}

    for comp in tickers:
        ll = 'df_'+comp
        reg = '^'+comp+'_'
        filtered_data = stock_data.filter(regex=reg)
        last_rows = loaded_data[ll].tail(filtered_data.shape[0])
        ind_ = filtered_data.index
        last_rows.index = ind_
        merged_df = filtered_data.join(last_rows, how='left', lsuffix='', rsuffix='_r')
        merged_df = merged_df[[col for col in merged_df.columns if not col.endswith('_r')]]
        inference_data[ll] = merged_df
            
    return inference_data

# %%
def lstm_predictions_vals(ticker):
    
    inference_data = data_loading()
    print(inference_data.keys())
    print("Loading data Completed")    
    new_model = load_model('lstm_model_general.h5')
    print("Loading LSTM model Completed")
    predictions = get_predictions_for_all_companies(inference_data, new_model,ticker, sequence_length=60)
    
    return predictions

# %%
#def main():
#    lstm_preds = lstm_predictions("AMZN")
#    return lstm_preds
