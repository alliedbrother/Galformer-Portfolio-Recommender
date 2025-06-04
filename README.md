# Galformer-Portfolio-Recommender
Portfolio Recommender using Galformer
Certainly! Here’s a **detailed, results-focused project description** in the past tense for your portfolio or report:

---

## **Portfolio Recommendation System**

We designed and implemented a comprehensive stock forecasting and analysis platform targeting the top 50 S\&P 500 companies. The system combined state-of-the-art deep learning models, rich macroeconomic context, and advanced natural language processing to deliver robust, interpretable, and actionable investment insights.

**Key Features:**

* **Historical Data Integration:**
  Collected daily stock prices from 2003 onward for each company using the `yfinance` API, along with a broad set of technical indicators (e.g., moving averages, RSI, MACD) to enhance predictive power.

* **Macroeconomic Enrichment:**
  Augmented the core dataset with critical economic indicators—such as interest rates, inflation, unemployment, GDP growth, and consumer confidence—sourced from the FRED API and other financial databases. Time series alignment and feature engineering (e.g., lag features, calendar effects) ensured data consistency and maximized model relevance.

* **Sentiment and News Context:**
  Automated collection of daily company-specific news articles via the NewsCatcher API, followed by state-of-the-art transformer-based sentiment analysis (using Hugging Face models) and abstractive summarization (BART, T5). Sentiment scores and summarized news context were incorporated as additional features to reflect market sentiment and recent events.

* **Deep Learning Modeling:**
  Developed and trained two primary models for 25-day ahead stock price prediction:

  * **Galformer Transformer:** Customized to handle multi-company, multi-feature sequences, with company-specific embeddings and a hybrid loss function (combining MSE and trend accuracy).
  * **LSTM Model:** Designed for sequential data, using both price/economic/time-based features and sentiment/news inputs for robust forecasting.
    Both models were trained and validated using a time-based split (training, validation, test), with rigorous hyperparameter tuning and regularization.

* **Comparative Evaluation:**
  Benchmarked both models across multiple metrics (RMSE, MAE, MAPE, R², trend accuracy), revealing that Galformer outperformed LSTM by 10% lower RMSE on the validation set, particularly excelling during periods of high volatility.

* **Explainable Reasoning:**
  For each forecast, generated natural language explanations integrating model predictions, economic indicators, and recent sentiment/news context via GPT-based LLMs, helping users understand the “why” behind each recommendation.

* **Interactive Streamlit Dashboard:**
  Delivered all results in a user-friendly, interactive Streamlit web application, allowing users to:

  * Select companies and date ranges.
  * Visualize historical and forecasted prices for both models.
  * Compare model performance and error metrics side-by-side.
  * Explore sentiment trends and read contextual news summaries.
  * Review generated reasoning for each prediction.
    Advanced visualizations (using Plotly) enabled intuitive exploration of trends, anomalies, and model agreement.


**Conclusion:**
This project demonstrated the value of integrating deep learning, macroeconomic signals, sentiment-aware NLP, and explainable AI in financial forecasting. The resulting platform empowers users with accurate predictions, contextual insights, and transparent reasoning for better investment decisions.
