# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf  # For fetching stock prices
import os
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import re
import urllib.parse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import requests
import json
from lstm_model_predictions_vals import lstm_predictions_vals


from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import feedparser

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Set page configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Define prediction horizon and sequence length
prediction_horizon = 25  # Number of days ahead to predict (updated to match models)
sequence_length = 60    # Sequence length expected by the models

# Ticker to company name mapping (same as before)
ticker_to_company_name = {
    "AAPL": "Apple Inc.",
    "NVDA": "NVIDIA Corporation",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "WMT": "Walmart Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "XOM": "Exxon Mobil Corporation",
    "UNH": "UnitedHealth Group Incorporated",
    "ORCL": "Oracle Corporation",
    "MA": "Mastercard Incorporated",
    "HD": "The Home Depot Inc.",
    "PG": "The Procter & Gamble Company",
    "COST": "Costco Wholesale Corporation",
    "JNJ": "Johnson & Johnson",
    "NFLX": "Netflix Inc.",
    "ABBV": "AbbVie Inc.",
    "BAC": "Bank of America Corporation",
    "KO": "The Coca-Cola Company",
    "CRM": "Salesforce Inc.",
    "CVX": "Chevron Corporation",
    "MRK": "Merck & Co. Inc.",
    "AMD": "Advanced Micro Devices Inc.",
    "PEP": "PepsiCo Inc.",
    "ACN": "Accenture plc",
    "LIN": "Linde plc",
    "MCD": "McDonald's Corporation",
    "CSCO": "Cisco Systems Inc.",
    "ADBE": "Adobe Inc.",
    "WFC": "Wells Fargo & Company",
    "IBM": "International Business Machines Corporation",
    "GE": "General Electric Company",
    "ABT": "Abbott Laboratories",
    "DHR": "Danaher Corporation",
    "AXP": "American Express Company",
    "MS": "Morgan Stanley",
    "CAT": "Caterpillar Inc.",
    "NOW": "ServiceNow Inc.",
    "QCOM": "QUALCOMM Incorporated",
    "PM": "Philip Morris International Inc.",
    "ISRG": "Intuitive Surgical Inc.",
    "VZ": "Verizon Communications Inc.",
}

# Fetch historical data function
def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        st.warning(f"No historical data available for {ticker}.")
    return data

# Prepare input sequence function (updated normalization)
def prepare_input_sequence(data, sequence_length, ticker):
    if 'Close' not in data.columns or len(data) < sequence_length:
        st.warning(f"Not enough data to create input sequence for {ticker}.")
        return None
    close_prices = data['Close'].values.astype(np.float32)

    # Prepare input features (for one feature 'Close')
    input_sequence = close_prices[-sequence_length:]

    # Normalize features (z-score standardization)
    mean = np.mean(input_sequence)
    std = np.std(input_sequence) + 1e-6  # To avoid division by zero
    normalized_sequence = (input_sequence - mean) / std

    # Reshape to match model input
    input_sequence = normalized_sequence.reshape(1, sequence_length, 1)
    return input_sequence

# Galformer predictions function (updated to match training preprocessing)
def galformer_predictions(ticker, sequence_length, model):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    data = fetch_historical_data(ticker, start_date, end_date)
    input_sequence = prepare_input_sequence(data, sequence_length, ticker)
    if input_sequence is not None:
        try:
            # For Galformer, add positional encoding if needed
            # Assuming the model includes positional encoding internally
            predictions = model.predict(input_sequence)
            # Denormalize predictions
            mean = np.mean(data['Close'].values[-prediction_horizon:])
            std = np.std(data['Close'].values[-prediction_horizon:]) + 1e-6
            predictions = predictions * std + mean
            return predictions.flatten()[:prediction_horizon]  # Return predictions for the prediction horizon
        except Exception as e:
            st.error(f"Error making Galformer predictions for {ticker}: {e}")
            logging.error(f"Error making Galformer predictions for {ticker}: {e}")
            return None
    else:
        return None

# LSTM predictions function (similar to Galformer)
def lstm_predictions(ticker, sequence_length, model):
    predictions = lstm_predictions_vals(ticker)
    print("*" * 25)
    predictions = predictions["df_"+ticker.upper()]
    print("*" * 25)
    return predictions

# Summarize articles function
def summarize_articles(articles, summarizer, summarizer_tokenizer):
    
    full_text = articles[0]

    if not full_text:
        return 'No summary available.'

    # Use regex to split sentences
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    max_input_length = summarizer_tokenizer.model_max_length

    # Split sentences into chunks whose tokenized length is within max_input_length
    chunks = []
    current_chunk = ''
    current_length = 0

    for sentence in sentences:
        sentence_length = len(summarizer_tokenizer.encode(sentence, add_special_tokens=False))
        if current_length + sentence_length <= max_input_length:
            current_chunk += ' ' + sentence
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    summaries = []

    # Process all chunks in batch
    try:
        batch_size = 8  # Adjust batch size as needed
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_summaries = summarizer(
                batch_chunks,
                max_length=100,
                min_length=30,
                do_sample=False,
                truncation=True  # Explicitly enable truncation
            )
            summaries.extend([summary['summary_text'] for summary in batch_summaries])
    except Exception as e:
        st.error(f"Error summarizing articles: {e}")
        logging.error(f"Error summarizing articles: {e}")
        return 'No summary available.'

    combined_summary = ' '.join(summaries)
    return combined_summary if combined_summary else 'No summary available.'

# Analyze sentiment summary function
def analyze_sentiment_summary(summary_text, sentiment_classifier):
    if summary_text:
        text = summary_text[:512]  # Truncate if necessary
        try:
            result = sentiment_classifier(text)[0]
            # Convert sentiment label to score
            label = result['label'].lower()
            if label == 'positive':
                score = result['score']
            elif label == 'negative':
                score = -result['score']
            else:
                score = 0  # Neutral sentiment
            return score
        except Exception as e:
            st.error(f"Error analyzing sentiment of summary: {e}")
            logging.error(f"Error analyzing sentiment of summary: {e}")
            return 0
    else:
        return 0

# Analyze article sentiments function
def analyze_article_sentiments(articles, sentiment_classifier):
    sentiment_data = []

    texts = []
    dates = []

    for article in articles:
        texts = article + ""

    try:
        batch_results = sentiment_classifier(texts)
        for result, date in zip(batch_results, dates):
            label = result['label'].lower()
            if label == 'positive':
                score = result['score']
            elif label == 'negative':
                score = -result['score']
            else:
                score = 0  # Neutral sentiment
            sentiment_data.append({'Date': date, 'Sentiment Score': score})
    except Exception as e:
        logging.error(f"Error analyzing sentiment for articles: {e}")
        st.error(f"Error analyzing sentiment for articles: {e}")
        return pd.DataFrame()

    sentiment_df = pd.DataFrame(sentiment_data)
    if sentiment_df.empty:
        st.warning("No sentiment data available to analyze.")
        return pd.DataFrame()
    daily_sentiment = sentiment_df.groupby('Date').mean().reset_index()
    return daily_sentiment

# Generate reasoning function
def generate_reasoning(ticker, current_price, predicted_prices, sentiment_score, text_generator):
    prompt = (
        f"As a seasoned financial analyst, provide a detailed analysis for {ticker} based on the following data:\n"
        f"- Current Price: ${current_price:.2f}\n"
        f"- Predicted Prices for the next {prediction_horizon} days: {predicted_prices}\n"
        f"- Sentiment Score based on recent news: {sentiment_score:.2f}\n\n"
        f"Please include market trends, potential risks, and an investment recommendation. Explain your reasoning thoroughly. Dont add in any references"
        f"and keep the ouput as plain text and your answer must cover : what the forecast says, what to do buy/sell/hold, what can we expect in the future"
        f"See to it that you maintain the same font type for everything be it numbers or regular text"
    )
    try:
        
        """generated = text_generator(
            prompt,
            max_length=1000,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            truncation=True  # Explicitly enable truncation
        )[0]['generated_text']
        reasoning = generated.strip()"""

        """generator = pipeline('text-generation', model='gpt-2')
        analysis = generator(prompt, max_length=500, num_return_sequences=1)
        reasoning = analysis.strip()"""

        url = "https://api.perplexity.ai/chat/completions"

        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "Be precise and concise."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        json_text = json.loads(response.text)['choices'][0]['message']['content']      
        reasoning = json_text.strip()

        reasoning = reasoning.encode('utf-8').decode('utf-8')

        # Ensure the reasoning ends gracefully
        if reasoning and reasoning[-1] not in ['.', '!', '?']:
            reasoning += '.'
        return reasoning
    except Exception as e:
        st.error(f"Error generating reasoning for {ticker}: {e}")
        logging.error(f"Error generating reasoning for {ticker}: {e}")
        return "Unable to generate reasoning at this time."

# Asynchronous function to fetch RSS feed
async def fetch_rss_feed(session, feed_url):
    async with session.get(feed_url) as response:
        return await response.text()

# Fetch Google News RSS function (asynchronous)


async def fetch_google_news_rss_async(ticker, max_articles):
    import requests
    url = "https://api.newscatcherapi.com/v2/search"
    comp_name = ticker_to_company_name[ticker]
    querystring = {"q":{comp_name},"lang":"en","sort_by":"relevancy","page":"1"}
    headers = {
        "x-api-key": "EEXiyyXLkDjFm6cnTNj4QxvXnpduY9xXn9E26IUYtOg"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    data = response.json()
    
    if data.get("status") == "ok" and "articles" in data:
        # Extract only the summaries
        summaries = []
        for article in data["articles"][:max_articles]:  # Limit to max_articles if needed
            summaries.append(article.get("summary"))
        
        return summaries
    else:
        print(f"Failed to fetch articles or no articles found for {comp_name}.")
        return []


# Function to fetch news for multiple tickers asynchronously
async def fetch_news_for_tickers(tickers, max_articles):
    tasks = []
    for ticker in tickers:
        tasks.append(fetch_google_news_rss_async(ticker, max_articles))
    results = await asyncio.gather(*tasks)
    return {tickers[i]: results[i] for i in range(len(tickers))}

# Synchronous wrapper for asynchronous function
def fetch_news(tickers, max_articles):
    return asyncio.run(fetch_news_for_tickers(tickers, max_articles))

# Plot predictions function
def plot_predictions(ticker, current_price, lstm_pred, galformer_pred):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=90)
    historical_data = fetch_historical_data(ticker, start_date, end_date)
    if historical_data.empty:
        st.warning(f"No historical data available for {ticker}.")
        return
    historical_prices = historical_data['Close'].reset_index()
    historical_prices.rename(columns={'Date': 'Date', 'Close': 'Price'}, inplace=True)
    historical_prices.rename(columns={historical_prices.columns[1]: 'Price'}, inplace=True)
    historical_prices['Type'] = 'Historical'

    # Ensure 'Date' is datetime.datetime
    historical_prices['Date'] = pd.to_datetime(historical_prices['Date'])

    # Create future dates
    last_date = historical_prices['Date'].iloc[-1]
    future_dates = []
    delta = 1
    while len(future_dates) < prediction_horizon:
        next_date = last_date + timedelta(days=delta)
        if next_date.weekday() < 5:  # Skip weekends
            future_dates.append(next_date)
        delta += 1

    # Prepare DataFrames for predictions
    df_lstm = pd.DataFrame({
        'Date': future_dates,
        'Price': lstm_pred,
        'Type': 'LSTM Prediction'
    })
    df_galformer = pd.DataFrame({
        'Date': future_dates,
        'Price': galformer_pred,
        'Type': 'Galformer Prediction'
    })

    # Combine all DataFrames
    df_all = pd.concat([historical_prices, df_lstm, df_galformer], ignore_index=True)
    df_all.sort_values('Date', inplace=True)

    # Plot using Plotly
    fig = px.line(df_all, x='Date', y='Price', color='Type',
                  title=f"{ticker} Historical and Predicted Prices")

    # Add vertical line at the prediction start
    prediction_start = future_dates[0] - timedelta(days=0.5)
    fig.add_vrect(
        x0=prediction_start,
        x1=prediction_start,
        fillcolor="gray",
        opacity=0.5,
        line_width=0
    )

    # Add annotation
    fig.add_annotation(
        x=prediction_start,
        y=df_all['Price'].max(),
        text="Prediction Start",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )

    # Adjust layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title_text='Data Type'
    )
    
    combined_prices_df = {'Date': future_dates,
        'LSTM Predictions': df_lstm['Price'],
        'Transformer Predictions': df_galformer['Price']}
    
    st.dataframe(combined_prices_df)

    st.plotly_chart(fig, use_container_width=True, key=f"predictions-{ticker}")

# Plot sentiment function
def plot_sentiment(sentiment_score, key):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            title={'text': "Sentiment Score"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
            },
        )
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# Display results in tabs function
def display_results_in_tabs(company):
    st.subheader(f"{company['Ticker']} - ${company['Current Price']:.2f}")
    #st.metric(label="Sentiment Score", value=round(company['Sentiment Score'], 2))
    st.markdown("#### Sentiment Score")
    st.metric(label="", value=round(company['Sentiment Score'], 2))

    tabs = st.tabs(["Predictions", "Sentiment Score", "Reasoning"])

    with tabs[0]:
        st.markdown("#### Predictions")
        if company['Galformer Prediction'] is not None:
            plot_predictions(
                company['Ticker'],
                company['Current Price'],
                company['LSTM Prediction'],
                company['Galformer Prediction']
            )
        else:
            st.write("Predictions are not available.")

    with tabs[1]:
        st.markdown("#### Sentiment Score")
        plot_sentiment(company['Sentiment Score'], key=f"sentiment-{company['Ticker']}")
        st.markdown("#### News Summary")
        with st.expander("Show News Summary"):
            st.write(company['Summary'])

    with tabs[2]:
        st.markdown("#### Generated Reasoning")
        st.markdown(company['Reasoning'])

    # """with tabs[3]:
    #     st.markdown("#### Sentiment Over Time")
    #     daily_sentiment = company.get('Daily Sentiment')
    #     if daily_sentiment is not None and not daily_sentiment.empty:
    #         # Plot sentiment over time
    #         fig = px.line(daily_sentiment, x='Date', y='Sentiment Score',
    #                       title='Sentiment Over Time', markers=True)
    #         st.plotly_chart(fig, use_container_width=True, key=f"sentiment-over-time-{company['Ticker']}")

    #         # Fetch historical stock prices matching sentiment dates
    #         start_date = daily_sentiment['Date'].min()
    #         end_date = daily_sentiment['Date'].max() + timedelta(days=1)  # Include the last day
    #         historical_data = fetch_historical_data(company['Ticker'], start_date, end_date)
    #         if historical_data.empty:
    #             st.warning(f"No historical data available for {company['Ticker']}.")
    #             return
    #         historical_prices = historical_data['Close'].reset_index()
    #         historical_prices['Date'] = pd.to_datetime(historical_prices['Date']).dt.date
    #         historical_prices.rename(columns={'Close': 'Price'}, inplace=True)

    #         # Merge sentiment and historical prices
    #         merged_df = pd.merge(
    #             daily_sentiment,
    #             historical_prices,
    #             on='Date',
    #             how='inner'
    #         )

    #         if merged_df.empty:
    #             st.write("No overlapping dates between sentiment data and stock prices.")
    #             return

    #         # Create dual-axis plot
    #         fig = make_subplots(specs=[[{"secondary_y": True}]])

    #         # Add sentiment trace
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=merged_df['Date'],
    #                 y=merged_df['Sentiment Score'],
    #                 name="Sentiment Score",
    #                 mode='lines+markers'
    #             ),
    #             secondary_y=False,
    #         )

    #         # Add price trace using 'Price' column
    #         if 'Price' in merged_df.columns:
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=merged_df['Date'],
    #                     y=merged_df['Price'],
    #                     name="Stock Price",
    #                     mode='lines+markers'
    #                 ),
    #                 secondary_y=True,
    #             )

    #         # Update layout
    #         fig.update_layout(
    #             title_text="Sentiment Score and Stock Price Over Time"
    #         )
    #         fig.update_xaxes(title_text="Date")
    #         fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
    #         fig.update_yaxes(title_text="Stock Price", secondary_y=True)

    #         st.plotly_chart(fig, use_container_width=True, key=f"stock-price-{company['Ticker']}")
    #     else:
    #         st.write("No sentiment data available for this company.")"""

@st.cache_resource
def load_models():
    sentiment_model_name = "ProsusAI/finbert"
    summarizer_model_name = 'sshleifer/distilbart-cnn-12-6'  # Smaller, faster model
    generator_model_name = 'distilgpt2'  # Smaller, faster model

    # Load sentiment classifier
    sentiment_classifier = pipeline('sentiment-analysis', model=sentiment_model_name, framework='pt')

    # Load summarizer
    summarizer = pipeline('summarization', model=summarizer_model_name, tokenizer=summarizer_model_name, framework='pt')
    summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)

    # Load text generator
    text_generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
    text_generator_model = AutoModelForCausalLM.from_pretrained(generator_model_name)

    # Explicitly set eos_token_id and pad_token_id
    eos_token_id = text_generator_tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = text_generator_tokenizer.pad_token_id
    if eos_token_id is None:
        eos_token_id = text_generator_tokenizer.sep_token_id
    if eos_token_id is None:
        eos_token_id = 50256  # Default for GPT-2 models

    # Set the model configurations
    text_generator_model.config.eos_token_id = eos_token_id
    text_generator_model.config.pad_token_id = eos_token_id

    # Initialize text generator pipeline
    text_generator = pipeline(
        'text-generation',
        model=text_generator_model,
        tokenizer=text_generator_tokenizer,
        framework='pt',
        pad_token_id=eos_token_id
    )

    return sentiment_classifier, summarizer, summarizer_tokenizer, text_generator

@st.cache_resource
def load_galformer_model():
    model_path = 'enhanced_stock_galformer_model.keras'
    if not os.path.exists(model_path):
        st.error(
            f"Galformer model file not found at '{model_path}'. Please ensure the model file is available."
        )
        logging.error(f"Galformer model file not found at '{model_path}'.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        # No need to compile model if it's already compiled
        return model
    except Exception as e:
        st.error(f"Error loading Galformer model: {e}")
        logging.error(f"Error loading Galformer model: {e}")
        return None

# Load LSTM model function
@st.cache_resource
def load_lstm_model():
    model_path = 'enhanced_stock_lstm_model.keras'
    if not os.path.exists(model_path):
        st.error(
            f"LSTM model file not found at '{model_path}'. Please ensure the model file is available."
        )
        logging.error(f"LSTM model file not found at '{model_path}'.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        # No need to compile model if it's already compiled
        return model
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")
        logging.error(f"Error loading LSTM model: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_current_prices(tickers):
    data = yf.download(tickers, period="1d", interval="1d", threads=True, progress=False)
    current_prices = {}
    for ticker in tickers:
        try:
            # Get the most recent closing price
            if isinstance(data['Close'], pd.Series):
                current_price = data['Close'].dropna()[-1]
            else:
                current_price = data['Close'][ticker].dropna()[-1]
            current_prices[ticker] = current_price
        except Exception as e:
            st.error(f"Error fetching price for {ticker}: {e}")
            logging.error(f"Error fetching price for {ticker}: {e}")
            current_prices[ticker] = np.nan  # Use NaN for missing data
    return current_prices

# Function to process each ticker (updated to use LSTM model)
# Adjusted process_ticker function to handle cases where LSTM predictions might be None
def process_ticker(ticker, current_price, articles, max_articles, sequence_length, prediction_horizon,
                   sentiment_classifier, summarizer, summarizer_tokenizer, text_generator, galformer_model, lstm_model):
    logging.info(f"Processing ticker {ticker}")
    company_info = {}
    try:
        if np.isnan(current_price):
            st.warning(f"Skipping {ticker} due to missing current price.")
            logging.warning(f"Skipping {ticker} due to missing current price.")
            return None

        # Get Galformer predictions
        galformer_pred = galformer_predictions(ticker, sequence_length, galformer_model)
        if galformer_pred is None:
            st.warning(f"Skipping {ticker} due to insufficient data for Galformer predictions.")
            logging.warning(f"Skipping {ticker} due to insufficient data for Galformer predictions.")
            return None
        galformer_pred = np.round(galformer_pred, 2)

        print('Galform preds: ', type(galformer_pred), len(galformer_pred), galformer_pred)

        # Get LSTM predictions (mocked)
        lstm_pred = lstm_predictions(ticker, sequence_length, lstm_model)
        if lstm_pred is None:
            st.warning(f"LSTM predictions unavailable for {ticker}, using Galformer predictions as a fallback.")
            logging.warning(f"LSTM predictions unavailable for {ticker}, using Galformer predictions as a fallback.")
            lstm_pred = galformer_pred  # Use Galformer predictions as a fallback
        else:
            lstm_pred = np.round(lstm_pred, 2)

        # Use the pre-fetched articles
        if not articles:
            st.warning(f"No news articles found for {ticker}.")
            logging.warning(f"No news articles found for {ticker}.")
            summary = 'No summary available.'
            sentiment = 0.0
            daily_sentiment = pd.DataFrame()
        else:
            # Summarize articles
            summary = summarize_articles(articles, summarizer, summarizer_tokenizer)
            # Perform sentiment analysis on summary
            sentiment = analyze_sentiment_summary(summary, sentiment_classifier)
            # Analyze sentiment over time
            daily_sentiment = analyze_article_sentiments(articles, sentiment_classifier)


        # Generate reasoning
        reasoning = generate_reasoning(
            ticker, current_price, galformer_pred.tolist(), sentiment, text_generator
        )

        # Assemble company information
        company_info = {
            'Ticker': ticker,
            'Current Price': current_price,
            'LSTM Prediction': lstm_pred.tolist(),
            'Galformer Prediction': galformer_pred.tolist(),
            'Sentiment Score': sentiment,
            'Summary': summary,
            'Reasoning': reasoning,
            'Daily Sentiment': daily_sentiment,
        }

        logging.info(f"Processed {ticker}")
        return company_info
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")
        st.error(f"Error processing {ticker}: {e}")
        return None
    
# Application Layout
st.title("Stock Analysis Dashboard")

# Sidebar for user inputs
st.sidebar.header("Configure Analysis")

# Ticker selection
all_tickers = list(ticker_to_company_name.keys())

selected_tickers = st.sidebar.multiselect(
    "Select Tickers", options=all_tickers, default=["TSLA"]
)

# Number of articles
max_articles = st.sidebar.slider(
    "Maximum Articles per Company", min_value=5, max_value=20, value=10, step=1
)

# Process button
if st.sidebar.button("Run Analysis"):
    with st.spinner("Loading models..."):
        sentiment_classifier, summarizer, summarizer_tokenizer, text_generator = load_models()
        galformer_model = load_galformer_model()
        lstm_model = load_lstm_model()
        if galformer_model is None or lstm_model is None:
            st.stop()

    with st.spinner("Fetching current stock prices..."):
        current_prices = fetch_current_prices(selected_tickers)

    with st.spinner("Fetching news articles..."):
        news_data = fetch_news(selected_tickers, max_articles)

    combined_data = []

    total_tickers = len(selected_tickers)
    progress_bar = st.progress(0)

    # Prepare thread pool executor
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks
        futures = {
            executor.submit(
                process_ticker,
                ticker,
                current_prices.get(ticker, np.nan),
                news_data.get(ticker, []),
                max_articles,
                sequence_length,
                prediction_horizon,
                sentiment_classifier,
                summarizer,
                summarizer_tokenizer,
                text_generator,
                galformer_model,
                lstm_model  # Pass the LSTM model
            ): ticker for ticker in selected_tickers
        }

        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            try:
                company_info = future.result()
                if company_info:
                    combined_data.append(company_info)
                    st.success(f"Processed {ticker}")
                else:
                    st.warning(f"Ticker {ticker} was skipped.")
            except Exception as e:
                st.error(f"Error processing {ticker}: {e}")
                logging.error(f"Error processing {ticker}: {e}")
            progress_bar.progress((i + 1) / total_tickers)

    # Display results
    st.header("Analysis Results")

    for company in combined_data:
        display_results_in_tabs(company)

    # Option to download results
    st.header("Download Results")

    prediction_records = []

    for company in combined_data:
        for i in range(len(company['Galformer Prediction'])):
            record = {
                'Ticker': company['Ticker'],
                'Current Price': company['Current Price'],
                'Day Ahead': i + 1,
                'LSTM Prediction': company['LSTM Prediction'][i],
                'Galformer Prediction': company['Galformer Prediction'][i],
                'Sentiment Score': company['Sentiment Score'],
                'Reasoning': company['Reasoning'],
                'Summary': company['Summary'],
            }
            prediction_records.append(record)

    predictions_df = pd.DataFrame(prediction_records)

    # Convert DataFrame to CSV
    csv = predictions_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name='combined_predictions.csv',
        mime='text/csv',
    )

else:
    st.info("Please configure analysis parameters and click 'Run Analysis'.")
