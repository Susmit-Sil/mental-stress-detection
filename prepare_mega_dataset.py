import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CREATING MEGA EMOTION DATASET - 4 DATASETS COMBINED")
print("="*70)

# ========== DATASET 1: Your Sentiment Dataset ==========
print("\n📁 Loading Dataset 1: Sentiment Dataset...")
try:
    df1 = pd.read_csv('sentimentdataset.csv')
    df1['text'] = df1['Text'].astype(str)
    df1['emotion'] = df1['Sentiment'].str.strip()
    df1 = df1[['text', 'emotion']]
    print(f"✅ Loaded {len(df1)} samples from sentimentdataset.csv")
except Exception as e:
    print(f"⚠️  Could not load sentimentdataset.csv: {e}")
    df1 = pd.DataFrame(columns=['text', 'emotion'])

# ========== DATASET 2: Your Combined Data (Anxiety) ==========
print("\n📁 Loading Dataset 2: Combined Data (Anxiety)...")
try:
    df2 = pd.read_csv('Combined Data.csv')
    
    # Check column names
    print(f"   Columns found: {df2.columns.tolist()}")
    
    # Try different possible column names
    text_col = None
    emotion_col = None
    
    # Find text column
    for col in df2.columns:
        col_lower = str(col).lower().strip()
        if 'statement' in col_lower or 'text' in col_lower:
            text_col = col
            break
    
    # Find emotion column
    for col in df2.columns:
        col_lower = str(col).lower().strip()
        if 'status' in col_lower or 'emotion' in col_lower or 'label' in col_lower:
            emotion_col = col
            break
    
    if text_col and emotion_col:
        df2['text'] = df2[text_col].astype(str)
        df2['emotion'] = df2[emotion_col].astype(str).str.strip()
        df2 = df2[['text', 'emotion']]
        print(f"✅ Loaded {len(df2)} samples from Combined Data.csv")
        print(f"   Using columns: {text_col} → {emotion_col}")
    else:
        print(f"⚠️  Could not find statement/status columns")
        df2 = pd.DataFrame(columns=['text', 'emotion'])
        
except Exception as e:
    print(f"⚠️  Could not load Combined Data.csv: {e}")
    df2 = pd.DataFrame(columns=['text', 'emotion'])

# ========== DATASET 3: Emotion Dataset (Hugging Face) ==========
print("\n📥 Downloading Dataset 3: Emotion Dataset (436K samples)...")
print("   This will take 2-3 minutes...")
try:
    dataset = load_dataset("dair-ai/emotion")
    
    # Convert to dataframe
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Combine all splits
    df3 = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Map numeric labels to emotion names
    emotion_map = {
        0: 'Sadness', 
        1: 'Joy', 
        2: 'Love', 
        3: 'Anger', 
        4: 'Fear', 
        5: 'Surprise'
    }
    df3['emotion'] = df3['label'].map(emotion_map)
    df3 = df3[['text', 'emotion']]
    
    print(f"✅ Downloaded {len(df3)} samples from Hugging Face")
except Exception as e:
    print(f"⚠️  Could not download Emotion dataset: {e}")
    df3 = pd.DataFrame(columns=['text', 'emotion'])

# ========== DATASET 4: GoEmotions Dataset ==========
print("\n📥 Downloading Dataset 4: GoEmotions (58K samples)...")
print("   This will take 1-2 minutes...")
try:
    goemotions = load_dataset("google-research-datasets/go_emotions", "simplified")
    
    df4 = pd.DataFrame(goemotions['train'])
    
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    def get_primary_emotion(labels):
        for i, val in enumerate(labels):
            if val == 1:
                return emotion_labels[i].capitalize()
        return 'Neutral'
    
    df4['emotion'] = df4['labels'].apply(get_primary_emotion)
    df4 = df4[['text', 'emotion']]
    
    print(f"✅ Downloaded {len(df4)} samples from GoEmotions")
except Exception as e:
    print(f"⚠️  Could not download GoEmotions: {e}")
    df4 = pd.DataFrame(columns=['text', 'emotion'])

# ========== COMBINE ALL DATASETS ==========
print("\n🔗 Combining all 4 datasets...")

all_datasets = []
dataset_info = []

if len(df1) > 0:
    all_datasets.append(df1)
    dataset_info.append(("Sentiment Dataset", len(df1)))

if len(df2) > 0:
    all_datasets.append(df2)
    dataset_info.append(("Combined Data (Anxiety)", len(df2)))

if len(df3) > 0:
    all_datasets.append(df3)
    dataset_info.append(("Emotion Dataset (HF)", len(df3)))

if len(df4) > 0:
    all_datasets.append(df4)
    dataset_info.append(("GoEmotions", len(df4)))

if len(all_datasets) == 0:
    print("❌ ERROR: No datasets loaded!")
    exit(1)

# Combine
df_mega = pd.concat(all_datasets, ignore_index=True)

# Clean data
print("\n🧹 Cleaning data...")
df_mega = df_mega.dropna()
df_mega['text'] = df_mega['text'].astype(str).str.strip()
df_mega['emotion'] = df_mega['emotion'].astype(str).str.strip()

# Remove empty or very short texts
df_mega = df_mega[df_mega['text'].str.len() >= 10]

# Remove emotions with less than 10 samples
emotion_counts = df_mega['emotion'].value_counts()
valid_emotions = emotion_counts[emotion_counts >= 10].index
df_mega = df_mega[df_mega['emotion'].isin(valid_emotions)]

print(f"\n✅ MEGA DATASET CREATED!")
print("="*70)
print(f"Total samples: {len(df_mega):,}")
print(f"Unique emotions: {df_mega['emotion'].nunique()}")

print("\n📊 Dataset Breakdown:")
for name, size in dataset_info:
    percentage = (size / len(df_mega)) * 100
    print(f"  {name:30s}: {size:>8,} samples ({percentage:>5.1f}%)")

# Show emotion distribution
print("\n📊 Top 20 Emotion Distribution:")
emotion_dist = df_mega['emotion'].value_counts().head(20)
for emotion, count in emotion_dist.items():
    print(f"  {emotion:20s}: {count:>6,}")

# Save mega dataset
df_mega.to_csv('mega_emotion_dataset.csv', index=False)
print(f"\n💾 Saved to: mega_emotion_dataset.csv")

# Create balanced version (max 1500 per emotion for speed)
print("\n⚖️  Creating balanced dataset...")
balanced_samples = []
for emotion in df_mega['emotion'].unique():
    emotion_df = df_mega[df_mega['emotion'] == emotion]
    sample_size = min(1500, len(emotion_df))
    balanced_samples.append(emotion_df.sample(n=sample_size, random_state=42))

df_balanced = pd.concat(balanced_samples, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

df_balanced.to_csv('balanced_emotion_dataset.csv', index=False)
print(f"✅ Balanced dataset: {len(df_balanced):,} samples")
print(f"   Saved to: balanced_emotion_dataset.csv")

# Show anxiety representation
anxiety_count = df_mega[df_mega['emotion'].str.lower().str.contains('anxiety|anxious')]['emotion'].value_counts()
if len(anxiety_count) > 0:
    print(f"\n🎯 ANXIETY DETECTION:")
    for emotion, count in anxiety_count.items():
        print(f"   {emotion}: {count:,} samples")

print("\n🎉 DATA PREPARATION COMPLETE!")
print("\n📦 Files created:")
print("  1. mega_emotion_dataset.csv (FULL - all samples)")
print("  2. balanced_emotion_dataset.csv (BALANCED - faster training)")
print("\n💡 Recommendation: Use balanced version for 30-min training")
