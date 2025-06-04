# %% [markdown]
# ### Import Libraries

# %%
import tensorflow as tf  # For TFRecords
from fredapi import Fred
import pandas as pd
import numpy as np
import glob
import holidays
import pandas_ta as ta
import math
import os
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
tf.config.threading.set_inter_op_parallelism_threads(8)  # Setting threads as per your system

# %% [markdown]
# ### Initialize FRED API

# %%
# Initialize Fred with your API key
fred = Fred(api_key='db61e0d65c4d2a1053221aec21822d4e')  # Replace with your actual API key securely

# Define date range
start_date = '2002-01-01'
end_date = '2024-11-28'

# Define the indicators and their series IDs
indicators = {
    'Effective Federal Funds Rate': 'FEDFUNDS',
    '10-Year Treasury Rate': 'DGS10',
    'Consumer Price Index': 'CPIAUCSL',
    'Producer Price Index': 'PPIACO',
    'Unemployment Rate': 'UNRATE',
    'Nonfarm Payroll Employment': 'PAYEMS',
    'Real GDP': 'GDPC1',
    'Housing Starts': 'HOUST',
    'Industrial Production Index': 'INDPRO',
    'M2 Money Stock': 'M2SL',
    'Crude Oil Prices': 'DCOILWTICO',
    'Retail Sales': 'RSXFS',
    'Total Business Inventories': 'BUSINV'
}

# Fetch the data with date range
economic_data = pd.DataFrame()

for name, series_id in indicators.items():
    try:
        data = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date
        )
        if data is not None and not data.empty:
            economic_data[name] = data
            print(f"Successfully fetched data for {name}")
    except Exception as e:
        print(f"Error fetching {name}: {e}")

# Convert index to datetime if not already
economic_data.index = pd.to_datetime(economic_data.index)

# Ensure the DataFrame is sorted by date
economic_data.sort_index(inplace=True)

# %% [markdown]
# ### Fill Missing Values in Economic Data

# %%
# Fill missing values
def fill_missing_values(df):
    df_filled = df.copy()

    # Ensure index is DatetimeIndex
    if not isinstance(df_filled.index, pd.DatetimeIndex):
        df_filled.index = pd.to_datetime(df_filled.index)
    df_filled.sort_index(inplace=True)

    # Create Month and Year columns once
    df_filled['Month'] = df_filled.index.month
    df_filled['Year'] = df_filled.index.year

    # Process each column individually
    for column in df.columns:
        col_data = df_filled[['Year', 'Month', column]].copy()

        # Calculate monthly means
        monthly_means = col_data.groupby(['Year', 'Month'])[column].mean().rename('Monthly_Mean').reset_index()

        # Merge monthly means back into col_data
        col_data = col_data.merge(monthly_means, on=['Year', 'Month'], how='left')

        # Fill missing values with Monthly Mean
        null_mask = col_data[column].isnull()
        col_data.loc[null_mask, column] = col_data.loc[null_mask, 'Monthly_Mean']

        # Calculate yearly means
        yearly_means = col_data.groupby('Year')[column].mean().rename('Yearly_Mean').reset_index()

        # Merge yearly means into col_data
        col_data = col_data.merge(yearly_means, on='Year', how='left')

        # Fill remaining missing values with Yearly Mean
        still_null_mask = col_data[column].isnull()
        col_data.loc[still_null_mask, column] = col_data.loc[still_null_mask, 'Yearly_Mean']

        # Update the filled values back into df_filled
        df_filled.loc[:, column] = col_data[column].values

    # Drop the auxiliary columns
    df_filled.drop(['Month', 'Year'], axis=1, inplace=True)

    return df_filled

# Apply the function to fill missing values
economic_data_filled = fill_missing_values(economic_data)
economic_data_filled.index = pd.to_datetime(economic_data_filled.index)
economic_data_filled.sort_index(inplace=True)

# %% [markdown]
# ### Read and Preprocess Stock Data

# %%
# Read stock data
stock_data = pd.read_parquet('sp500_50stocks_data.parquet')
# Convert index to DatetimeIndex if not already
stock_data.index = pd.to_datetime(stock_data.index)
# Sort by date
stock_data.sort_index(inplace=True)

# Flatten MultiIndex columns in stock_data
stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

# Merge economic data with stock data
economic_data_filled.index.name = 'Date'
stock_data.index.name = 'Date'

# Create a daily date range based on stock data index
daily_date_range = pd.date_range(
    start=stock_data.index.min(),
    end=stock_data.index.max(),
    freq='D'  # Daily frequency
)

# Reindex economic data to daily frequency using forward fill
economic_data_daily = economic_data_filled.reindex(daily_date_range, method='ffill')
economic_data_daily.index.name = 'Date'

# Merge the DataFrames using the date index
combined_data = stock_data.join(economic_data_daily, how='left')
combined_data.ffill(inplace=True)

# %% [markdown]
# ### Additional Feature Engineering

# %%
tickers = ["AAPL", "NVDA", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "LLY", "TSLA",
           "WMT", "JPM", "V", "XOM", "UNH", "ORCL", "MA", "HD", "PG", "COST", "JNJ",
           "NFLX", "ABBV", "BAC", "KO", "CRM", "CVX", "MRK", "TMUS", "AMD", "PEP",
           "ACN", "LIN", "TMO", "MCD", "CSCO", "ADBE", "WFC", "IBM", "GE", "ABT",
           "DHR", "AXP", "MS", "CAT", "NOW", "QCOM", "PM", "ISRG", "VZ"]

print("Initial number of columns:", len(combined_data.columns))

for ticker in tickers:
    close_col = f'{ticker}_Close'

    if close_col in combined_data.columns:
        # Calculate technical indicators
        combined_data[f'{ticker}_SMA_20'] = ta.sma(combined_data[close_col], length=20)
        combined_data[f'{ticker}_RSI_14'] = ta.rsi(combined_data[close_col], length=14)
        macd = ta.macd(combined_data[close_col], fast=12, slow=26)
        macd_columns = [f'{ticker}_MACD', f'{ticker}_MACD_Hist', f'{ticker}_MACD_Signal']
        macd.columns = macd_columns
        combined_data = pd.concat([combined_data, macd], axis=1)
        bbands = ta.bbands(combined_data[close_col], length=20)
        bbands_columns = [f'{ticker}_BB_Lower', f'{ticker}_BB_Middle', f'{ticker}_BB_Upper', f'{ticker}_BB_Bandwidth', f'{ticker}_BB_Percentage']
        bbands.columns = bbands_columns
        combined_data = pd.concat([combined_data, bbands], axis=1)
        combined_data[f'{ticker}_MOM_10'] = ta.mom(combined_data[close_col], length=10)
    else:
        print(f'Column {close_col} not found in combined_data.')

print("Number of columns after adding technical indicators:", len(combined_data.columns))

# %% [markdown]
# ### Create Lag Features

# %%
# Create lag features
n_lags = 5
lagged_features = {}

# Create lag features for stock prices
for ticker in tickers:
    close_col = f'{ticker}_Close'

    if close_col in combined_data.columns:
        for lag in range(1, n_lags + 1):
            lag_col_name = f'{ticker}_Close_Lag_{lag}'
            lagged_features[lag_col_name] = combined_data[close_col].shift(lag)
    else:
        print(f'Column {close_col} not found in combined_data.')

# Create lag features for economic indicators
economic_indicators = economic_data_filled.columns.tolist()

for indicator in economic_indicators:
    if indicator in combined_data.columns:
        for lag in range(1, n_lags + 1):
            lag_col_name = f'{indicator}_Lag_{lag}'
            lagged_features[lag_col_name] = combined_data[indicator].shift(lag)
    else:
        print(f'Indicator {indicator} not found in combined_data.')

# Convert the lagged features dictionary to a DataFrame
lagged_features_df = pd.DataFrame(lagged_features, index=combined_data.index)

# Concatenate the lagged features DataFrame to the original DataFrame
combined_data = pd.concat([combined_data, lagged_features_df], axis=1)
combined_data = combined_data.copy()

print(f"Total number of columns after adding lag features: {len(combined_data.columns)}")

# %% [markdown]
# ### Extract Date-Based Features

# %%
# Extract Date-Based Features
# Extract day of the week
combined_data['Day_of_Week'] = combined_data.index.dayofweek  # Monday=0, Sunday=6

# Extract month
combined_data['Month'] = combined_data.index.month

# Extract quarter
combined_data['Quarter'] = combined_data.index.quarter

# Identify US holidays
us_holidays = holidays.US()
combined_data['Is_Holiday'] = combined_data.index.isin(us_holidays).astype(int)

# Identify month start and end
combined_data['Is_Month_Start'] = combined_data.index.is_month_start.astype(int)
combined_data['Is_Month_End'] = combined_data.index.is_month_end.astype(int)

# %% [markdown]
# ### Split Combined Data into Individual Company DataFrames

# %%
# Split combined data into individual company DataFrames
all_dfs = {}  # Dictionary to store DataFrames

for tick in tickers:
    # Create a dynamic DataFrame name
    df_name = "df_" + tick

    # Filter the combined_data DataFrame for columns matching the ticker
    company_df = combined_data.filter(like=tick, axis=1)

    # Add date-based features and economic indicators
    company_df = pd.concat([
        company_df,
        combined_data[['Day_of_Week', 'Month', 'Quarter', 'Is_Holiday', 'Is_Month_Start', 'Is_Month_End']],
        combined_data[economic_indicators]
    ], axis=1)

    # Add target variables
    for i in range(1, 6):  # Forwarded columns for 1 to 5 steps
        company_df[f'{tick}_target_{i}'] = company_df[f'{tick}_Close'].shift(-i)

    # Drop missing values
    company_df.dropna(inplace=True)

    # Store the DataFrame in the dictionary
    all_dfs[df_name] = company_df

# Adjust DataFrames for specific companies if necessary (as in your original code)
filtered_list = ['Day_of_Week', 'Month', 'Quarter', 'Is_Holiday', 'Is_Month_Start', 'Is_Month_End'] + economic_indicators

# Adjust DataFrames for specific companies
special_companies = ['df_MA', 'df_MS', 'df_V', 'df_PM', 'df_GOOG']
for comp in special_companies:
    ticker = comp.split('_')[1]
    all_dfs[comp] = all_dfs[comp].loc[:, all_dfs[comp].columns.str.startswith(f'{ticker}_') | all_dfs[comp].columns.isin(filtered_list)]
    print(f"{comp} shape after adjustment: {all_dfs[comp].shape}")

# %% [markdown]
# ### Prepare Data for Models and Serialize Using TFRecords

# %%
# Function to prepare data sequences
def prepare_sequence_data(df, sequence_length=60, prediction_horizon=5):
    X, y = [], []

    # Ensure the DataFrame is sorted by date
    df = df.sort_index()

    # Select relevant input features (exclude targets)
    input_features = df.filter(regex="^(?!.*target).*").values
    targets = df.filter(regex="target").values

    # Create sequences
    for i in range(len(df) - sequence_length - prediction_horizon + 1):
        seq_x = input_features[i : i + sequence_length]
        seq_y = targets[i + sequence_length : i + sequence_length + prediction_horizon]
        X.append(seq_x)
        y.append(seq_y.flatten())

    return np.array(X), np.array(y)

# Prepare data for all companies
sequence_length = 60  # Length of input sequences
prediction_horizon = 5  # Number of days to predict

X_list, y_list = [], []
for company, df in all_dfs.items():
    print(f"Preparing data for {company}...")
    X_company, y_company = prepare_sequence_data(df, sequence_length, prediction_horizon)
    X_list.append(X_company)
    y_list.append(y_company)

# Concatenate data from all companies
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print(f"Final shape of X: {X.shape}")
print(f"Final shape of y: {y.shape}")

# %% [markdown]
# #### **Convert Data to float32**

# %%
# Convert X and y to float32
X = X.astype(np.float32)
y = y.astype(np.float32)

print(f"Data types after conversion: X - {X.dtype}, y - {y.dtype}")

# %% [markdown]
# ### Save the Preprocessed Data into TFRecord Files

# %%

# Define the number of shards (TFRecord files)
num_shards = 10  # You can adjust this number based on your needs

# Calculate the number of samples per shard
num_samples = X.shape[0]
shard_size = math.ceil(num_samples / num_shards)

# Create a directory to store TFRecord files
tfrecord_dir = 'tfrecords_data'
os.makedirs(tfrecord_dir, exist_ok=True)

# Function to serialize features and labels
def _bytes_feature(value):
    """Returns a bytes_list from a string/byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is a tensor
        value = value.numpy()  # get the numpy value
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(feature, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Ensure feature and label are in float32
    feature = feature.astype(np.float32)
    label = label.astype(np.float32)

    # Serialize the feature and label tensors
    feature_serialized = tf.io.serialize_tensor(feature)
    label_serialized = tf.io.serialize_tensor(label)

    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type
    feature_dict = {
        'feature': _bytes_feature(feature_serialized.numpy()),
        'label': _bytes_feature(label_serialized.numpy()),
    }

    # Create a Features message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()

# Write data to TFRecord files
for shard_id in range(num_shards):
    start_idx = shard_id * shard_size
    end_idx = min(start_idx + shard_size, num_samples)

    shard_X = X[start_idx:end_idx]
    shard_y = y[start_idx:end_idx]

    tfrecord_filename = os.path.join(tfrecord_dir, f'shard_{shard_id}.tfrecord')
    with tf.io.TFRecordWriter(tfrecord_filename, options='GZIP') as writer:
        for i in range(shard_X.shape[0]):
            example = serialize_example(shard_X[i], shard_y[i])
            writer.write(example)
    print(f"TFRecord shard {shard_id} written with {end_idx - start_idx} samples.")

print("All TFRecord files have been written.")
