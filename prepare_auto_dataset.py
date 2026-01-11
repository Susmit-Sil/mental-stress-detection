import pandas as pd
import os
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("AUTO DATASET PREPARATION - SMART COLUMN DETECTION")
print("="*70)

# ========== STEP 1: Auto-detect local CSV/Excel files ==========
print("\n🔍 Scanning folder for dataset files...")

local_datasets = []
supported_extensions = ['.csv', '.xlsx', '.xls', '.tsv']

# Find all dataset files in current folder
for file in os.listdir('.'):
    if any(file.endswith(ext) for ext in supported_extensions):
        local_datasets.append(file)

if local_datasets:
    print(f"✅ Found {len(local_datasets)} local dataset(s):")
    for i, file in enumerate(local_datasets, 1):
        print(f"   {i}. {file}")
else:
    print("⚠️  No CSV/Excel files found in current folder")

# ========== STEP 2: Load and auto-detect columns ==========
all_data = []

for file in local_datasets:
    print(f"\n📁 Processing: {file}")
    
    try:
        # Load file (auto-detect CSV or Excel)
        if file.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.endswith('.tsv'):
            df = pd.read_csv(file, sep='\t')
        else:  # Excel
            df = pd.read_excel(file)
        
        print(f"   Loaded {len(df)} rows")
        print(f"   Columns: {df.columns.tolist()}")
        
        # ========== AUTO-DETECT TEXT COLUMN ==========
        text_col = None
        possible_text_names = [
            'text', 'statement', 'sentence', 'content', 'message', 
            'review', 'comment', 'post', 'tweet', 'description'
        ]
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(name in col_lower for name in possible_text_names):
                text_col = col
                break
        
        # If not found, use first column with long text
        if not text_col:
            for col in df.columns:
                if df[col].dtype == 'object':
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 20:  # Likely text
                        text_col = col
                        break
        
        # ========== AUTO-DETECT EMOTION/LABEL COLUMN ==========
        emotion_col = None
        possible_emotion_names = [
            'emotion', 'sentiment', 'label', 'status', 'feeling',
            'mood', 'class', 'category', 'target'
        ]
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(name in col_lower for name in possible_emotion_names):
                emotion_col = col
                break
        
        # If not found, use column with few unique values (likely labels)
        if not emotion_col:
            for col in df.columns:
                if col != text_col and df[col].dtype == 'object':
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.1:  # Less than 10% unique = likely labels
                        emotion_col = col
                        break
        
        # ========== VALIDATE AND ADD ==========
        if text_col and emotion_col:
            print(f"   ✅ Detected:")
            print(f"      Text column: '{text_col}'")
            print(f"      Emotion column: '{emotion_col}'")
            
            # Extract and standardize
            df_clean = pd.DataFrame({
                'text': df[text_col].astype(str),
                'emotion': df[emotion_col].astype(str).str.strip()
            })
            
            # Remove missing values
            df_clean = df_clean.dropna()
            
            all_data.append(df_clean)
            print(f"   ✅ Added {len(df_clean)} valid samples")
        else:
            print(f"   ⚠️  Could not auto-detect columns. Skipping.")
            print(f"      Found text column: {text_col}")
            print(f"      Found emotion column: {emotion_col}")
            
    except Exception as e:
        print(f"   ❌ Error processing {file}: {e}")

# ========== STEP 3: Download online datasets (optional) ==========
print("\n" + "="*70)
download_online = input("Download online datasets (Emotion + GoEmotions)? (yes/no): ").strip().lower()

if download_online in ['yes', 'y']:
    # Emotion Dataset
    print("\n📥 Downloading Emotion Dataset (436K samples)...")
    try:
        dataset = load_dataset("dair-ai/emotion")
        df_emotion = pd.concat([
            pd.DataFrame(dataset['train']),
            pd.DataFrame(dataset['validation']),
            pd.DataFrame(dataset['test'])
        ])
        emotion_map = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
        df_emotion['emotion'] = df_emotion['label'].map(emotion_map)
        df_emotion = df_emotion[['text', 'emotion']]
        all_data.append(df_emotion)
        print(f"✅ Added {len(df_emotion):,} samples from Emotion Dataset")
    except Exception as e:
        print(f"⚠️  Could not download Emotion Dataset: {e}")
    
    # GoEmotions
    print("\n📥 Downloading GoEmotions (58K samples)...")
    try:
        goemotions = load_dataset("google-research-datasets/go_emotions", "simplified")
        df_go = pd.DataFrame(goemotions['train'])
        
        emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        
        def get_emotion(labels):
            for i, val in enumerate(labels):
                if val == 1:
                    return emotion_labels[i].capitalize()
            return 'Neutral'
        
        df_go['emotion'] = df_go['labels'].apply(get_emotion)
        df_go = df_go[['text', 'emotion']]
        all_data.append(df_go)
        print(f"✅ Added {len(df_go):,} samples from GoEmotions")
    except Exception as e:
        print(f"⚠️  Could not download GoEmotions: {e}")

# ========== STEP 4: Combine and save ==========
if not all_data:
    print("\n❌ No datasets loaded! Please add CSV/Excel files to the folder.")
    exit(1)

print("\n" + "="*70)
print("🔗 Combining all datasets...")

df_combined = pd.concat(all_data, ignore_index=True)

# Clean
df_combined = df_combined.dropna()
df_combined['text'] = df_combined['text'].astype(str).str.strip()
df_combined['emotion'] = df_combined['emotion'].astype(str).str.strip()
df_combined = df_combined[df_combined['text'].str.len() >= 10]

# Remove rare emotions
emotion_counts = df_combined['emotion'].value_counts()
valid_emotions = emotion_counts[emotion_counts >= 10].index
df_combined = df_combined[df_combined['emotion'].isin(valid_emotions)]

print(f"\n✅ COMBINED DATASET CREATED!")
print(f"   Total samples: {len(df_combined):,}")
print(f"   Unique emotions: {df_combined['emotion'].nunique()}")

print("\n📊 Top 15 Emotions:")
for emotion, count in df_combined['emotion'].value_counts().head(15).items():
    print(f"   {emotion:20s}: {count:>6,}")

# Save
df_combined.to_csv('auto_combined_dataset.csv', index=False)
print(f"\n💾 Saved to: auto_combined_dataset.csv")

# Create balanced version
print("\n⚖️  Creating balanced version...")
balanced = []
for emotion in df_combined['emotion'].unique():
    emotion_df = df_combined[df_combined['emotion'] == emotion]
    sample_size = min(1500, len(emotion_df))
    balanced.append(emotion_df.sample(n=sample_size, random_state=42))

df_balanced = pd.concat(balanced, ignore_index=True).sample(frac=1, random_state=42)
df_balanced.to_csv('auto_balanced_dataset.csv', index=False)
print(f"✅ Balanced: {len(df_balanced):,} samples")

print("\n" + "="*70)
print("🎉 DONE! Your datasets are ready for training!")
print("\nNext step: python train_auto_model.py")
