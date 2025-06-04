
1. **Economic Indicators:**
   - Effective Federal Funds Rate
   - 10-Year Treasury Rate
   - Consumer Price Index
   - Producer Price Index
   - Unemployment Rate
   - Nonfarm Payroll Employment
   - Real GDP
   - Housing Starts
   - Industrial Production Index
   - M2 Money Stock
   - Crude Oil Prices
   - Retail Sales
   - Total Business Inventories

2. **Stock Data Features for Each Ticker:**
   - Close prices for each of the 50 tickers.
   - Technical indicators for each stock:
     - Simple Moving Average (SMA) over 20 days.
     - Relative Strength Index (RSI) over 14 days.
     - Moving Average Convergence Divergence (MACD), MACD Histogram, and MACD Signal.
     - Bollinger Bands: Lower, Middle, Upper, Bandwidth, and Percentage.
     - Momentum over 10 days.

3. **Lag Features:**
   - Lagged values for stock prices and economic indicators for up to 5 days.

4. **Date-Based Features:**
   - Day of the week (0 for Monday, 6 for Sunday).
   - Month of the year.
   - Quarter of the year.
   - Whether the day is a U.S. holiday.
   - Whether the day is the start of the month.
   - Whether the day is the end of the month.

5. **Target Variables:**
   - Future close prices (targets) for each stock, shifted by 1 to 5 days forward.

6. **Normalized and Scaled:**
   - The model training process includes normalization (z-score standardization) of the features and labels. 

7. **Unique step for Galformer - Positional Encoding:**
   - The Galformer model introduces positional encoding, which is a common technique in Transformer models to give the model information about the position of each item in the sequence. This is not an additional feature from the dataset but rather a transformation applied to the input features to enhance the model's ability to learn temporal dependencies.
