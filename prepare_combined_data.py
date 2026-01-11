import pandas as pd
import numpy as np

print("="*60)
print("COMBINING DATASETS FOR MENTAL STRESS DETECTION")
print("="*60)

# Load sentiment dataset
print("\n📁 Loading sentiment dataset...")
df_sentiment = pd.read_csv('sentimentdataset.csv')
df_sentiment['Text'] = df_sentiment['Text'].astype(str)
df_sentiment['Sentiment'] = df_sentiment['Sentiment'].str.strip()

print(f"✅ Loaded {len(df_sentiment)} samples from sentiment dataset")
print(f"   Unique emotions: {df_sentiment['Sentiment'].nunique()}")

# Load hotel reviews dataset
print("\n📁 Loading hotel reviews dataset...")
df_hotel = pd.read_csv('Datafiniti_Hotel_Reviews.csv')
df_hotel = df_hotel.dropna(subset=['reviews.text', 'reviews.rating'])

print(f"✅ Loaded {len(df_hotel)} samples from hotel reviews")

# Map hotel ratings to emotions that match sentiment dataset
def rating_to_emotion(rating):
    """Map hotel ratings to mental health related emotions"""
    if rating >= 4.5:
        return 'Joy'  # Very positive experience
    elif rating >= 4.0:
        return 'Positive'  # Good experience
    elif rating >= 3.5:
        return 'Contentment'  # Satisfied
    elif rating >= 3.0:
        return 'Neutral'  # Okay experience
    elif rating >= 2.0:
        return 'Disappointment'  # Not satisfied
    elif rating >= 1.5:
        return 'Frustration'  # Bad experience
    else:
        return 'Negative'  # Very bad experience

df_hotel['Sentiment'] = df_hotel['reviews.rating'].apply(rating_to_emotion)
df_hotel['Text'] = df_hotel['reviews.text'].astype(str)

# Keep only text and sentiment columns
df_sentiment_clean = df_sentiment[['Text', 'Sentiment']].copy()
df_hotel_clean = df_hotel[['Text', 'Sentiment']].copy()

# Combine both datasets
print("\n🔗 Combining datasets...")
df_combined = pd.concat([df_sentiment_clean, df_hotel_clean], ignore_index=True)

print(f"\n✅ COMBINED DATASET CREATED!")
print(f"   Total samples: {len(df_combined)}")
print(f"   From sentiment: {len(df_sentiment_clean)}")
print(f"   From hotel: {len(df_hotel_clean)}")
print(f"   Unique emotions: {df_combined['Sentiment'].nunique()}")

# Show emotion distribution
print("\n📊 Top 15 Emotion Distribution:")
print(df_combined['Sentiment'].value_counts().head(15))

# Save combined dataset
df_combined.to_csv('combined_dataset.csv', index=False)
print("\n💾 Saved to: combined_dataset.csv")

# Show some examples
print("\n📝 Sample texts from combined dataset:")
print("\nExample 1 (from sentiment):")
print(f"Text: {df_combined.iloc[0]['Text'][:100]}...")
print(f"Emotion: {df_combined.iloc[0]['Sentiment']}")

print("\nExample 2 (from hotel):")
print(f"Text: {df_combined.iloc[800]['Text'][:100]}...")
print(f"Emotion: {df_combined.iloc[800]['Sentiment']}")

print("\n✅ Data preparation complete!")
