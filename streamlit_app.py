import requests
import streamlit as st
from datetime import datetime
from transformers import pipeline
import time
import pandas as pd  
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# === Streamlit Page Setup ===
st.set_page_config(page_title="NYT Stock Sentiment Analyzer", layout="centered")
st.title("📰 NYT Stock Sentiment Analyzer with FinBERT")

# === Load FinBERT Model ===
@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_model():
    try:
        import torch
        # Explicitly set device to CPU if GPU not available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        
        # Try with more explicit parameters
        return pipeline(
            task="text-classification", 
            model="ProsusAI/finbert",
            device=device,
            framework="pt"  # Explicitly choose PyTorch
        )
    except Exception as e:
        st.warning(f"Error loading model with standard pipeline: {e}")
        
        try:
            # Alternative: Load model components separately
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            
            # Create a custom pipeline function
            def custom_finbert(text):
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                score = probabilities[0][prediction].item()
                
                labels = {0: "positive", 1: "negative", 2: "neutral"}
                return [{"label": labels[prediction], "score": score}]
            
            return custom_finbert
        except Exception as e2:
            st.error(f"Failed to load FinBERT model: {e2}")
            
            # Provide a simple fallback
            def simple_sentiment(text):
                # Very basic sentiment analysis as fallback
                positive_words = ["increase", "higher", "rise", "up", "growth", "profit", "gain"]
                negative_words = ["decrease", "lower", "fall", "down", "decline", "loss"]
                
                text_lower = text.lower()
                pos_count = sum(word in text_lower for word in positive_words)
                neg_count = sum(word in text_lower for word in negative_words)
                
                if pos_count > neg_count:
                    return [{"label": "positive", "score": 0.75}]
                elif neg_count > pos_count:
                    return [{"label": "negative", "score": 0.75}]
                else:
                    return [{"label": "neutral", "score": 0.8}]
                
            return simple_sentiment


# === Sidebar Inputs ===
st.sidebar.header("Search Settings")
ticker = st.sidebar.text_input("Stock Ticker", "META").upper()
company_name = st.sidebar.text_input("Company Name", "Meta Platforms")
start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2023, 12, 31))
api_key = "TpS1Gyw9HqMBjYw5aXiPdLjMwOwGoSAA"  # Replace with your own key
max_pages = st.sidebar.slider("Max Pages to Fetch", 1, 10, 3)
run_analysis = st.sidebar.button("Analyze")

# === NYT API Fetch Function ===
def fetch_nyt_articles(query, begin_date, end_date, api_key, page=0):
    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    fq = f"news_desk:(\"Business\" \"Financial\") AND ({query})"
    params = {
        'q': query,
        'fq': fq,
        'begin_date': begin_date.replace("-", ""),
        'end_date': end_date.replace("-", ""),
        'sort': 'relevance',
        'page': page,
        'api-key': api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Request error: {e}")
        return None

# === Historical Stock Fetch ===
@st.cache_data(show_spinner="Fetching stock prices...")
def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]
    
    df = df[["Date", "Close"]]
    
    return df

# === Run the App Logic ===
if run_analysis:
    st.subheader(f"Searching NYT for `{ticker}` ({company_name})")
    query = company_name if company_name else ticker

    all_articles = []
    processed_ids = set()

    with st.spinner("Fetching articles from NYT..."):
        for page in range(max_pages):
            st.write(f"📄 Fetching page {page + 1}...")
            data = fetch_nyt_articles(query, start_date.isoformat(), end_date.isoformat(), api_key, page)
            
            if not data or 'response' not in data or 'docs' not in data['response']:
                st.warning("No more articles or error occurred.")
                break
            
            docs = data['response']['docs']
            if not docs:
                break

            all_articles.extend(docs)
            time.sleep(12)  # Respect NYT rate limit

    st.success(f"Fetched {len(all_articles)} articles.")

    # === Sentiment Analysis ===
    total_score = 0
    num_articles = 0
    sentiment_rows = []

    st.subheader("🧠 Sentiment Analysis Results")
    for article in all_articles:
        _id = article['_id']
        if _id in processed_ids:
            continue
        processed_ids.add(_id)

        headline = article['headline'].get('main', '')
        lead = article.get('lead_paragraph', '')
        full_text = f"{headline} {lead}".strip()

        if not full_text:
            continue

      # Replace the current sentiment analysis code
try:
    # Try the direct approach first
    sentiment_result = pipe(full_text)
    # Check if result is a list and not empty
    if isinstance(sentiment_result, list) and sentiment_result:
        sentiment = sentiment_result[0]
    else:
        # Handle unexpected output format
        sentiment = {"label": "neutral", "score": 0.5}
except Exception as e:
    st.warning(f"Error analyzing sentiment: {e}")
    # Provide a fallback sentiment
    sentiment = {"label": "neutral", "score": 0.5}
        label, score = sentiment['label'], sentiment['score']
        date = article['pub_date'][:10]
        url = article.get('web_url', '')

        sentiment_rows.append({
            "Date": date,
            "Headline": headline,
            "Sentiment": label,
            "Score": round(score, 2),
            "URL": url
        })

        with st.expander(f"{headline}"):
            st.write(f"📅 **Date:** {date}")
            st.write(f"🔗 [View Article]({url})")
            st.write(f"💬 **Sentiment:** `{label}` (Score: `{score:.2f}`)")

        if label == 'positive':
            total_score += score
            num_articles += 1
        elif label == 'negative':
            total_score -= score
            num_articles += 1

    # === Final Sentiment Summary ===
    final_avg = total_score / num_articles if num_articles > 0 else 0
    sentiment_summary = "Positive" if final_avg > 0.15 else "Negative" if final_avg < -0.15 else "Neutral"

    st.markdown("---")
    st.subheader("📊 Overall Sentiment")
    st.markdown(f"""
    > **Sentiment:** {sentiment_summary}  
    > **Average Score:** `{final_avg:.2f}`
    """)

    # === Sentiment Table View ===
    st.markdown("### 🗂️ Sentiment Table")
    df_sentiment = pd.DataFrame(sentiment_rows)
    st.dataframe(df_sentiment, use_container_width=True)

    # === Historical Stock Prices ===
    st.markdown("### 💹 Historical Stock Prices")
    stock_df = fetch_stock_data(ticker, start_date, end_date)

    if stock_df.empty:
        st.warning("⚠️ No stock data found for this period. Check the ticker or date range.")
    else:
        st.success(f"Fetched {len(stock_df)} stock price records.")

        # Display stock price table
        st.markdown("#### Historical Stock Price Data")
        stock_table = stock_df.copy()
        stock_table["Date"] = stock_table["Date"].dt.strftime('%Y-%m-%d')
        stock_table["Close"] = stock_table["Close"].round(2).apply(lambda x: f"${x:.2f}")
        st.dataframe(stock_table, use_container_width=True)

        # Preprocess for plotting
        stock_df = stock_df.reset_index()
        df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"])
        stock_df["Date"] = pd.to_datetime(stock_df["Date"])

        # Group sentiment by date and map score
        daily_sentiment = df_sentiment.groupby("Date")["Score"].mean().reset_index()
        daily_sentiment.rename(columns={"Score": "Sentiment"}, inplace=True)

        # Merge with stock prices
        merged_df = pd.merge(stock_df, daily_sentiment, on="Date", how="left")
        merged_df = merged_df.fillna(method="ffill")  # Forward fill missing sentiment

        # Plot Close and Sentiment
        merged_df.set_index("Date", inplace=True)
        st.line_chart(merged_df[["Close", "Sentiment"]], use_container_width=True)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Continue only if stock_df exists
if not stock_df.empty:
    st.markdown("---")
    st.subheader("🤖 LSTM Stock Price Prediction")

    # Normalize the Close prices
    close_prices = stock_df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = scaler.fit_transform(close_prices)

    # Prepare sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    sequence_length = 60
    X, y = create_sequences(close_scaled, sequence_length)

    # Split into train/test
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # Reshape for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    with st.spinner("Training LSTM model..."):
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(monitor='loss', patience=3)])

    st.success("Model training complete!")

    # Make predictions
    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Prepare plot DataFrame
    test_dates = stock_df['Date'][sequence_length + split_idx:].reset_index(drop=True)
    lstm_df = pd.DataFrame({
        'Date': test_dates,
        'Actual Price': real_prices.flatten(),
        'Predicted Price': predicted_prices.flatten()
    })
    lstm_df.set_index('Date', inplace=True)

    st.line_chart(lstm_df, use_container_width=True)
    
    # LSTM Prediction Table
    st.markdown("#### LSTM Prediction Results")
    prediction_table = pd.DataFrame({
        'Date': test_dates,
        'Predicted Close Price': predicted_prices.flatten().round(2),
        'Actual Close Price': real_prices.flatten().round(2),
        'Difference': (real_prices.flatten() - predicted_prices.flatten()).round(2)
    })
    prediction_table['Date'] = prediction_table['Date'].dt.strftime('%Y-%m-%d')
    prediction_table['Predicted Close Price'] = prediction_table['Predicted Close Price'].apply(lambda x: f"${x:.2f}")
    prediction_table['Actual Close Price'] = prediction_table['Actual Close Price'].apply(lambda x: f"${x:.2f}")
    prediction_table['Difference'] = prediction_table['Difference'].apply(lambda x: f"${x:.2f}")
    st.dataframe(prediction_table, use_container_width=True)

    st.markdown("### 🔮 Next-Day Prediction")
    last_60_days = close_scaled[-sequence_length:]
    last_60_days = np.reshape(last_60_days, (1, sequence_length, 1))
    next_day_pred = model.predict(last_60_days)
    next_day_price = scaler.inverse_transform(next_day_pred)[0][0]

    st.metric(label="Predicted Next-Day Closing Price", value=f"${next_day_price:.2f}")

    # === EXTRA VISUALIZATIONS ===
    st.markdown("## 📈 Additional Visualizations")

    # --- Sentiment Distribution Pie Chart ---
    st.markdown("### 🥧 Sentiment Distribution (Pie Chart)")
    sentiment_counts = df_sentiment["Sentiment"].value_counts()
    st.pyplot(sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, figsize=(5, 5), ylabel="").get_figure())

    # --- Sentiment by Day (Bar Graph) ---
    st.markdown("### 📆 Average Daily Sentiment (Bar Graph)")
    st.bar_chart(daily_sentiment.set_index("Date")["Sentiment"])

    # Add sentiment interpretation instead of legend
    # Add sentiment interpretation for FinBERT
st.markdown("""
**Sentiment Interpretation:**
- When sentiment is **positive**, higher scores (closer to 1.0) indicate <span style='color:green'>stronger positive sentiment</span>
- When sentiment is **neutral**, higher scores indicate stronger neutral sentiment 
- When sentiment is **negative**, higher scores (closer to 1.0) indicate <span style='color:red'>stronger negative sentiment</span>

FinBERT classifies text into one of these three categories and provides a confidence score for that classification.
""", unsafe_allow_html=True)

# Monthly sentiment summary
st.markdown("### 📊 Monthly Sentiment Summary")
# Add month column for grouping
df_sentiment['Month'] = pd.to_datetime(df_sentiment['Date']).dt.strftime('%Y-%m')

# Group by month and sentiment, then count occurrences
monthly_sentiment_counts = df_sentiment.groupby(['Month', 'Sentiment']).size().unstack(fill_value=0)

# Calculate the dominant sentiment for each month
monthly_dominant = {}
for month in monthly_sentiment_counts.index:
    row = monthly_sentiment_counts.loc[month]
    dominant = row.idxmax()
    count = row[dominant]
    total = row.sum()
    percentage = (count / total) * 100
    monthly_dominant[month] = (dominant, percentage)

# Calculate average scores by sentiment type for each month
monthly_scores = df_sentiment.groupby(['Month', 'Sentiment'])['Score'].mean().reset_index()

# Create summary text based on monthly sentiment
monthly_summary = []
for month in sorted(monthly_dominant.keys()):
    dominant_sentiment, percentage = monthly_dominant[month]
    
    # Get average scores for each sentiment type in this month
    month_data = monthly_scores[monthly_scores['Month'] == month]
    score_text = []
    
    for sentiment in ['positive', 'neutral', 'negative']:
        sent_data = month_data[month_data['Sentiment'] == sentiment]
        if not sent_data.empty:
            avg_score = sent_data['Score'].values[0]
            count = monthly_sentiment_counts.loc[month, sentiment]
            
            if sentiment == 'positive':
                score_text.append(f"<span style='color:green'>{sentiment}</span> ({count} articles, avg score: {avg_score:.2f})")
            elif sentiment == 'negative':
                score_text.append(f"<span style='color:red'>{sentiment}</span> ({count} articles, avg score: {avg_score:.2f})")
            else:
                score_text.append(f"{sentiment} ({count} articles, avg score: {avg_score:.2f})")
    
    # Format the dominant sentiment with color
    if dominant_sentiment == 'positive':
        dom_text = f"<span style='color:green'>{dominant_sentiment}</span>"
    elif dominant_sentiment == 'negative':
        dom_text = f"<span style='color:red'>{dominant_sentiment}</span>"
    else:
        dom_text = dominant_sentiment
        
    sentiment_text = f"<li>In {month}, the dominant sentiment was {dom_text} ({percentage:.1f}% of articles)<br>Breakdown: {', '.join(score_text)}</li>"
    monthly_summary.append(sentiment_text)

st.markdown("<ul>" + "".join(monthly_summary) + "</ul>", unsafe_allow_html=True)

# --- Volume of Articles Over Time (Improved) ---
st.markdown("### 📰 Number of Articles Per Day")

# Group by date to get count of articles
article_volume = df_sentiment.groupby(pd.to_datetime(df_sentiment["Date"]).dt.date).size().reset_index()
article_volume.columns = ["Date", "Article Count"]

# Convert to dates for proper plotting
article_volume["Date"] = pd.to_datetime(article_volume["Date"])

# Create a nice looking plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=article_volume, x="Date", y="Article Count", marker='o', linewidth=2, color='royalblue')
plt.title("Article Coverage Over Time", fontsize=16)
plt.ylabel("Number of Articles", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
st.pyplot(fig)

# Also show as table
st.markdown("#### Daily Article Count")
article_volume["Date"] = article_volume["Date"].dt.strftime('%Y-%m-%d')
st.dataframe(article_volume, use_container_width=True)

# === Final Verdict and Explanation ===
st.markdown("---")
st.subheader("🧩 Final Analysis Verdict")

if not stock_df.empty and len(prediction_table) > 0:
    # Calculate mean absolute percentage error to evaluate prediction accuracy
    actual_prices = real_prices.flatten()
    predicted_prices_flat = predicted_prices.flatten()
    
    # Calculate percentage error for each prediction
    percentage_errors = np.abs((actual_prices - predicted_prices_flat) / actual_prices) * 100
    mean_error = np.mean(percentage_errors)
    
    # Determine if our prediction is accurate
    if mean_error < 5:
        accuracy_assessment = "highly accurate"
        accuracy_color = "green"
    elif mean_error < 10:
        accuracy_assessment = "reasonably accurate"
        accuracy_color = "orange"
    else:
        accuracy_assessment = "potentially inaccurate"
        accuracy_color = "red"
    
    # Display prediction accuracy assessment
    st.markdown(f"""
    ### Prediction Accuracy Assessment
    
    Based on historical prediction performance, the model is <span style='color:{accuracy_color};font-weight:bold'>{accuracy_assessment}</span> with an average error of {mean_error:.2f}%.
    """, unsafe_allow_html=True)
    
    # Calculate overall sentiment trend
    if len(df_sentiment) > 0:
        positive_count = (df_sentiment['Sentiment'] == 'positive').sum()
        negative_count = (df_sentiment['Sentiment'] == 'negative').sum()
        neutral_count = (df_sentiment['Sentiment'] == 'neutral').sum()
        
        if positive_count > negative_count and positive_count > neutral_count:
            sentiment_trend = "predominantly positive"
            expected_price_movement = "likely to rise"
        elif negative_count > positive_count and negative_count > neutral_count:
            sentiment_trend = "predominantly negative"
            expected_price_movement = "likely to fall"
        else:
            sentiment_trend = "mostly neutral"
            expected_price_movement = "likely to remain stable"
        
        st.markdown(f"""
        ### Media Sentiment and Price Movement Correlation
        
        The media sentiment during this period was **{sentiment_trend}**, suggesting prices were {expected_price_movement} based on news coverage alone.
        """)
    
    # Check if there were significant price movements
    if len(stock_df) > 10:
        # Calculate daily returns
        stock_df_copy = stock_df.copy()
        stock_df_copy['Daily_Return'] = stock_df_copy['Close'].pct_change() * 100
        
        # Find significant price movements (more than 5% in a day)
        significant_movements = stock_df_copy[abs(stock_df_copy['Daily_Return']) > 5]
        
        if len(significant_movements) > 0:
            st.markdown("### Significant Stock Price Movements Detected")
            st.markdown(f"Found {len(significant_movements)} days with price movements greater than 5%:")
            
            for idx, row in significant_movements.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d')
                movement = "increase" if row['Daily_Return'] > 0 else "decrease"
                st.markdown(f"- **{date_str}**: {abs(row['Daily_Return']):.2f}% {movement}")
            
            # Directly research explanations for significant movements without a button
            from bs4 import BeautifulSoup
            
            st.markdown("### Price Movement Explanations")
            with st.spinner("Analyzing price movement causes..."):
                for idx, row in significant_movements.iterrows():
                    date_str = row['Date'].strftime('%Y-%m-%d')
                    date_obj = row['Date']
                    
                    # Format the date for better search queries (e.g., "May 3")
                    formatted_date = date_obj.strftime('%B %d').replace(' 0', ' ')
                    
                    movement = "increase" if row['Daily_Return'] > 0 else "decrease"
                    percent_change = abs(row['Daily_Return'])
                    
                    # Improved search query with more specific terms and company name
                    search_query = f"{ticker} {company_name} stock {percent_change:.1f}% {movement} {formatted_date} {date_obj.year} news reason why"
                    
                    try:
                        # Using requests to search without API
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
                        response = requests.get(search_url, headers=headers)
                        
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # Extract search results
                            search_results = []
                            for result in soup.select('div.g'):
                                # Get title
                                title_elem = result.select_one('h3')
                                if not title_elem:
                                    continue
                                
                                # Get snippet
                                snippet_elem = result.select_one('div.IsZvec')
                                if not snippet_elem:
                                    # Try alternative selector patterns
                                    snippet_elem = result.select_one('div.VwiC3b')
                                
                                snippet = snippet_elem.get_text() if snippet_elem else "No snippet available"
                                
                                search_results.append({
                                    'title': title_elem.get_text(),
                                    'snippet': snippet
                                })
                            
                            # Try alternative selector if no results found
                            if not search_results:
                                for result in soup.select('div.tF2Cxc'):
                                    title_elem = result.select_one('h3')
                                    snippet_elem = result.select_one('div.IsZvec, div.VwiC3b')
                                    
                                    if title_elem:
                                        snippet = snippet_elem.get_text() if snippet_elem else "No snippet available"
                                        search_results.append({
                                            'title': title_elem.get_text(),
                                            'snippet': snippet
                                        })
                            
                            if search_results:
                                st.markdown(f"#### Analysis for {date_str} ({abs(row['Daily_Return']):.2f}% {movement})")
                                
                                # Combine all snippets for analysis
                                combined_text = " ".join([r['snippet'] for r in search_results[:3]])
                                
                                # Basic NLP analysis to extract key phrases
                                import re
                                import nltk
                                
                                try:
                                    nltk.data.find('tokenizers/punkt')
                                except LookupError:
                                    nltk.download('punkt', quiet=True)
                                    nltk.download('stopwords', quiet=True)
                                
                                from nltk.corpus import stopwords
                                from nltk.tokenize import word_tokenize, sent_tokenize
                                
                                stop_words = set(stopwords.words('english'))
                                
                                # Tokenize and clean text
                                sentences = sent_tokenize(combined_text)
                                
                                # Find most relevant sentences
                                relevant_sentences = []
                                
                                # Add ticker and company name variations
                                search_terms = [
                                    ticker.lower(), 
                                    company_name.lower(), 
                                    'stock', 
                                    'shares', 
                                    'price', 
                                    movement.lower(), 
                                    'percent', 
                                    'earnings',
                                    'report',
                                    'announced'
                                ]
                                
                                for sentence in sentences:
                                    sentence_lower = sentence.lower()
                                    # Check if any search term is in the sentence
                                    if any(term in sentence_lower for term in search_terms):
                                        relevant_sentences.append(sentence)
                                
                                if relevant_sentences:
                                    st.markdown("Possible explanations found:")
                                    for i, sentence in enumerate(relevant_sentences[:3]):
                                        st.markdown(f"- {sentence}")
                                else:
                                    # If no relevant sentences found, show the first search result title and snippet
                                    if search_results:
                                        st.markdown("Related news found:")
                                        for i, result in enumerate(search_results[:2]):
                                            st.markdown(f"- **{result['title']}**: {result['snippet'][:200]}...")
                                    else:
                                        st.markdown("No clear explanation found in search results.")
                            else:
                                # Try an alternative search query
                                alt_search_query = f"{ticker} stock news {formatted_date} {date_obj.year}"
                                alt_search_url = f"https://www.google.com/search?q={alt_search_query.replace(' ', '+')}"
                                
                                try:
                                    alt_response = requests.get(alt_search_url, headers=headers)
                                    if alt_response.status_code == 200:
                                        alt_soup = BeautifulSoup(alt_response.text, 'html.parser')
                                        
                                        alt_search_results = []
                                        for result in alt_soup.select('div.g, div.tF2Cxc'):
                                            title_elem = result.select_one('h3')
                                            if not title_elem:
                                                continue
                                                
                                            snippet_elem = result.select_one('div.IsZvec, div.VwiC3b')
                                            snippet = snippet_elem.get_text() if snippet_elem else "No snippet available"
                                            
                                            alt_search_results.append({
                                                'title': title_elem.get_text(),
                                                'snippet': snippet
                                            })
                                            
                                        if alt_search_results:
                                            st.markdown(f"#### Analysis for {date_str} ({abs(row['Daily_Return']):.2f}% {movement})")
                                            st.markdown("General news around this date:")
                                            for i, result in enumerate(alt_search_results[:2]):
                                                st.markdown(f"- **{result['title']}**")
                                        else:
                                            st.warning(f"No search results found for {date_str} movement.")
                                    else:
                                        st.warning(f"No search results found for {date_str} movement.")
                                except Exception as e:
                                    st.warning(f"No search results found for {date_str} movement.")
                        else:
                            st.error(f"Failed to retrieve search results for {date_str} movement.")
                    except Exception as e:
                        st.error(f"Error researching {date_str} movement: {str(e)}")

    # Final investment recommendation
    if 'next_day_price' in locals():
        last_price = stock_df['Close'].iloc[-1]
        price_change = next_day_price - last_price
        percent_change = (price_change / last_price) * 100
        
        st.markdown("## 💰 Investment Recommendation")
        
        if percent_change > 2:
            recommendation = "BUY"
            color = "green"
            explanation = f"The model predicts a significant price increase of {percent_change:.2f}%, and the overall sentiment is {sentiment_trend}."
        elif percent_change < -2:
            recommendation = "SELL"
            color = "red"
            explanation = f"The model predicts a significant price decrease of {percent_change:.2f}%, and the overall sentiment is {sentiment_trend}."
        else:
            recommendation = "HOLD"
            color = "orange"
            explanation = f"The model predicts a minor price change of {percent_change:.2f}%, and the overall sentiment is {sentiment_trend}."
        
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
            <h2 style="text-align: center; color: {color};">{recommendation}</h2>
            <p style="text-align: center;">{explanation}</p>
            <p style="text-align: center; font-size: 0.8em;">
                <i>This is for educational purposes only. Not financial advice.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add RMSE and MSE error metrics in tabular format
    st.markdown("## 📊 Error Metrics")
    
    # Calculate errors
    mse = np.mean((actual_prices - predicted_prices_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_prices - predicted_prices_flat))
    
    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Mean Absolute Percentage Error (MAPE)'],
        'Value': [mse, rmse, mae, mean_error],
        'Description': [
            'Average of squared differences between predictions and actual values',
            'Square root of MSE, provides error in the same units as the data',
            'Average of absolute differences between predictions and actual values',
            'Average percentage difference between predictions and actual values'
        ]
    })
    
    # Display metrics table
    st.table(metrics_df)
    
    # Display predicted vs actual values in a table
    st.markdown("### Predicted vs Actual Values")
    
    # Create a comparison DataFrame
    comparison_df = pd.DataFrame({
        'Date': prediction_table['Date'],
        'Actual Price': actual_prices,
        'Predicted Price': predicted_prices_flat,
        'Absolute Error': np.abs(actual_prices - predicted_prices_flat),
        'Percentage Error': percentage_errors
    })
    
    # Format the values
    comparison_df['Actual Price'] = comparison_df['Actual Price'].map('${:,.2f}'.format)
    comparison_df['Predicted Price'] = comparison_df['Predicted Price'].map('${:,.2f}'.format)
    comparison_df['Absolute Error'] = comparison_df['Absolute Error'].map('${:,.2f}'.format)
    comparison_df['Percentage Error'] = comparison_df['Percentage Error'].map('{:,.2f}%'.format)
    
    # Display the comparison table
    st.dataframe(comparison_df)
