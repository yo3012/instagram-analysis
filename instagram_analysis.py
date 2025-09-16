# 🚀 Instagram Research Analysis (Single Page Project)
# Run: python instagram_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# ========== 1. Load Data ==========
def load_data(filepath):
    return pd.read_csv(filepath)

# ========== 2. Preprocess ==========
def preprocess_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['likes'] = df['likes'].fillna(0)
    df['comments'] = df['comments'].fillna(0)
    return df

# ========== 3. Sentiment Analysis ==========
def sentiment_analysis(df):
    def get_sentiment(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0
    df['sentiment'] = df['caption'].apply(get_sentiment)
    return df

# ========== 4. Visualizations ==========
def plot_engagement(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="likes", y="comments", data=df, hue="sentiment", palette="coolwarm")
    plt.title("📊 Engagement: Likes vs Comments")
    plt.show()

def plot_sentiment(df):
    plt.figure(figsize=(8,6))
    sns.histplot(df['sentiment'], bins=20, kde=True, color="skyblue")
    plt.title("💬 Sentiment Distribution of Captions")
    plt.show()

def top_hashtags(df):
    hashtags = []
    for row in df['hashtags'].dropna():
        hashtags.extend(row.replace("'", "").replace("[", "").replace("]", "").split(","))
    hashtags = [h.strip().lower() for h in hashtags if h.strip()]
    freq = pd.Series(hashtags).value_counts().head(10)
    
    plt.figure(figsize=(8,6))
    sns.barplot(x=freq.values, y=freq.index, palette="viridis")
    plt.title("🏷️ Top 10 Hashtags")
    plt.show()

# ========== 5. Main ==========
if __name__ == "__main__":
    print("🚀 Instagram Research Analysis Started")

    # Load sample CSV
    df = load_data("instagram_sample.csv")
    df = preprocess_data(df)
    df = sentiment_analysis(df)

    # Show Visualizations
    plot_engagement(df)
    plot_sentiment(df)
    top_hashtags(df)

    print("🎉 Analysis Complete!")
