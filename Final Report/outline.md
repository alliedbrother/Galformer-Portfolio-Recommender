### **1. Title Page**

- **Project Title**: *AI-Powered Portfolio Recommendation System Using Galformer*
- **Authors**: Bandham Manikanta, Sai Akhil Kogilathota, Aman Arya
- **Course Information**
- **Institution**
- **Date of Submission**

### **2. Abstract**

- A concise summary of the project, including the aims, methods, key results, and conclusions.

### **3. Table of Contents**

- A list of all main sections and sub-sections with corresponding page numbers.

### **4. Introduction**

- **Background**
  - Importance of portfolio recommendation systems in finance.
  - Evolution from traditional methods to AI-powered systems.
- **Problem Statement**
  - Challenges in accurate stock price prediction.
  - Limitations of existing models (e.g., LSTM) in modeling complex dependencies.
- **Objectives**
  - Develop a robust stock prediction model using Galformer.
  - Compare its performance with LSTM and existing tools.
  - Incorporate real-time news analysis for enhanced investment recommendations.
- **Scope of the Project**
  - Focus on top 50 S&P 500 companies from 2003 to the present.
  - Integration of macroeconomic indicators and technical analysis.
- **Report Structure**
  - Brief outline of how the report is organized.

### **5. Literature Review**

- **Traditional Stock Prediction Methods**
  - Overview of statistical models (e.g., ARIMA, linear regression).
- **Neural Networks in Finance**
  - Introduction to RNNs and their applications.
- **Long Short-Term Memory (LSTM) Networks**
  - Mechanism of LSTM networks.
  - Strengths in capturing short to medium-term dependencies.
  - Limitations in modeling long-term dependencies and complex patterns.
- **Transformer Models in Time-Series Forecasting**
  - Introduction to Transformers and their advantages over RNNs.
- **Galformer Model**
  - Detailed explanation of Galformer and its tailored design for time-series data.
  - Innovations over standard Transformer models.
- **Existing Tools for Portfolio Recommendation**
  - Overview of popular tools and platforms (e.g., Bloomberg Terminal, Quantopian, Personal Capital).
  - Features and limitations of these tools.
- **Comparison with Existing Tools**
  - How current project goals align or differ from existing solutions.
  - Gaps in existing tools that this project aims to address.
- **Sentiment Analysis in Stock Prediction**
  - The impact of news sentiment on stock prices.
  - Use of specialized NLP models like FinBERT in financial contexts.
- **Text Summarization for Financial News**
  - Importance of summarizing large volumes of news data.
  - Use of models like DistilBART for efficient summarization.
- **Investment Recommendation Systems**
  - Role of AI in generating investment advice.
  - Integration of predictive models and NLP.

### **6. Methodology**

#### **6.1. Overview of the Workflow**

- A visual diagram and explanation of the project's end-to-end process:
  - Data collection.
  - Feature engineering.
  - Model training (LSTM and Galformer).
  - Real-time news fetching and analysis.
  - Prediction and recommendation generation.
  - Dashboard development and rendering.

#### **6.2. Data Collection**

- **Financial Data**
  - Sources of stock price data for the top 50 S&P 500 companies.
  - Timeframe covered (2003 to present).
- **Macroeconomic Indicators**
  - Data on GDP, CPI, and other relevant economic indicators.
  - Sources and methods of acquisition.
- **Technical Indicators**
  - List of technical indicators used (e.g., RSI, MACD).
  - Rationale for inclusion.

#### **6.3. Data Preprocessing and Feature Engineering**

- **Handling Missing Values**
  - Techniques used (e.g., forward-filling).
- **Creating Lag Features**
  - Incorporation of past data (up to 5 days lag) as features.
- **Date-based Features**
  - Encoding temporal information (day of the week, month).
- **Normalization and Scaling**
  - Methods applied to standardize data.
- **Feature Selection**
  - Criteria for selecting relevant features.

#### **6.4. Model Development**

##### **6.4.1. LSTM Model**

- **Architecture Design**
  - Layer configuration.
  - Activation functions and other hyperparameters.
- **Training Process**
  - Data splits (training, validation, testing).
  - Loss function and optimization strategy.
- **Evaluation Metrics**
  - Use of Mean Absolute Error (MAE).
  - Justification for metric selection.
- **Limitations Identified**
  - Observations on the LSTM's performance challenges.

##### **6.4.2. Galformer Model**

- **Introduction to Galformer**
  - How Galformer extends the Transformer architecture for time-series data.
- **Key Innovations**
  - Enhanced attention mechanisms.
  - Improved positional encoding tailored for temporal sequences.
- **Architecture Details**
  - Comparison with regular Transformer architecture.
  - Explanation of specific layers and components.
- **Explainability of Galformer**
  - **Importance of Explainability in Financial Models**
    - Regulatory requirements.
    - Trust and adoption by users.
  - **Model Interpretability Techniques**
    - Attention mechanisms and their interpretability.
    - Visualization of attention weights to understand feature importance.
  - **Results of Explainability Analysis**
    - Insights gained from interpreting the model.
    - How explainability improves decision-making.
- **Training Process**
  - Hyperparameter tuning.
  - Regularization techniques (dropout, L2 regularization).
- **Performance Metrics**
  - Achieved MAE and other relevant evaluations.
- **Comparative Analysis**
  - How Galformer addresses LSTM's limitations.

#### **6.5. Real-time News Analysis and Recommendation Generation**

- **News Data Acquisition**
  - Sources for real-time financial news (e.g., RSS feeds, APIs).
- **Sentiment Analysis Using FinBERT**
  - Process of analyzing news sentiment.
  - Integration into the prediction system.
- **News Summarization with DistilBART**
  - Techniques for summarizing articles.
- **Investment Reasoning and Recommendations**
  - Use of Perplexity AI for generating detailed recommendations.
  - Methodology for combining predictions and news analysis.

#### **6.6. Dashboard Development**

- **Design Objectives**
  - User-friendly interface for investors.
  - Real-time data visualization.
- **Technologies Employed**
  - Frameworks and tools used (e.g., React, Flask).
- **Features Implemented**
  - Display of stock predictions.
  - News summaries and sentiment scores.
  - Personalized investment recommendations.
- **Integration Steps**
  - Connecting the backend models to the frontend interface.
- **Testing and Deployment**
  - Ensuring reliability and responsiveness.

### **7. Results**

#### **7.1. Model Performance**

- **LSTM Model Results**
  - Quantitative results (MAE of 1.8).
  - Graphs showing predicted vs. actual prices.
- **Galformer Model Results**
  - Improved quantitative results (MAE of 0.7).
  - Visual comparisons with LSTM results.
- **Comparison with Existing Tools**
  - Performance benchmarks against established tools.
  - Discussion of metrics used for comparison.
- **Analysis of Improvement**
  - Discussion on how Galformer's features contribute to better performance.

#### **7.2. Explainability Results**

- **Visualization of Attention Mechanisms**
  - Heatmaps or graphs showcasing attention weights.
- **Interpretation of Model Decisions**
  - Examples where explainability helped understand predictions.
- **Impact on Trust and Usability**
  - How explainability features enhance user confidence.

#### **7.3. Real-time News Analysis Outcomes**

- **Sentiment Analysis**
  - Aggregate sentiment scores for selected stocks.
  - Examples of sentiment impact on predictions.
- **News Summaries**
  - Sample summaries generated by DistilBART.
- **Investment Recommendations**
  - Examples of detailed recommendations provided to users.

#### **7.4. Dashboard Presentation**

- **User Interface Screenshots**
  - Visual representation of key dashboard features.
- **User Experience Highlights**
  - How users interact with the system.
- **Feedback and Observations**
  - Any user testing conducted and insights gained.

### **8. Discussion**

- **Interpretation of Findings**
  - Significance of Galformer's improved accuracy.
  - The impact of integrating technical and macroeconomic features.
- **Role of Explainability**
  - Importance in finance for compliance and user trust.
  - How explainability contributes to model adoption.
- **Comparison with Existing Tools**
  - Advantages over existing portfolio recommendation systems.
  - Features unique to your system.
  - Areas where existing tools may still have an edge.
- **Role of Real-time News Analysis**
  - Enhancement of predictions with sentiment data.
  - Added value for investors through summarization and recommendations.
- **Limitations and Challenges**
  - Data limitations (e.g., data quality, coverage).
  - Computational complexities.
  - Potential overfitting and generalization.

### **9. Conclusion**

- **Summary of Achievements**
  - Meeting the project objectives.
  - Demonstrated superiority of Galformer over LSTM and existing tools for this task.
- **Implications for the Industry**
  - Potential for AI models to enhance financial decision-making.
  - Importance of explainability for real-world applications.
- **Final Remarks**
  - Reflections on the project outcomes.

### **10. Future Work**

- **Model Enhancements**
  - Exploring other advanced architectures or ensemble methods.
- **Improving Explainability**
  - Implementing other interpretability techniques (e.g., SHAP values, LIME).
  - User studies on the effectiveness of explainability features.
- **Expansion of Features**
  - Inclusion of social media sentiment, alternative data sources.
- **Scalability**
  - Adapting the system for more markets or asset classes.
- **User Personalization**
  - Tailoring recommendations based on user profiles.
- **Automation and Real-world Deployment**
  - Steps towards integrating with brokerage platforms or financial services.

### **11. References**

- Comprehensive list of all academic papers, articles, models, and data sources referenced throughout the report, formatted according to a standard citation style (e.g., APA, IEEE).

### **12. Appendices**

- **Appendix A: Technical Indicator Calculations**
  - Definitions and formulas used for RSI, MACD, etc.
- **Appendix B: Model Hyperparameters**
  - Detailed listing of all hyperparameter settings for both LSTM and Galformer.
- **Appendix C: Additional Figures and Tables**
  - Supplementary charts, graphs, and statistical tables.
- **Appendix D: Code Excerpts**
  - Critical code snippets for key functions or processes.
- **Appendix E: Dashboard User Guide**
  - Instructions and tips for navigating and utilizing the dashboard features.

---
