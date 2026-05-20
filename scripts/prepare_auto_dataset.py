import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUT_DIR = os.path.join(BASE_DIR, 'data')
COMBINED_OUT = os.path.join(OUT_DIR, 'auto_combined_dataset.csv')
BALANCED_OUT = os.path.join(OUT_DIR, 'auto_balanced_dataset.csv')

SKIP_FILES = {
    'auto_combined_dataset.csv',
    'auto_balanced_dataset.csv',
}

SKIP_PATTERNS = [
    'hotel', 'airbnb', 'yelp_review', 'datafiniti',
]

TEXT_HINTS = ['text', 'statement', 'sentence', 'content', 'message',
              'review', 'comment', 'post', 'tweet', 'description', 'body']
EMOTION_HINTS = ['emotion', 'sentiment', 'label', 'status', 'feeling',
                 'mood', 'class', 'category', 'target', 'tag']

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def should_skip(filename: str) -> bool:
    name_lower = filename.lower()
    if filename in SKIP_FILES:
        return True
    if any(pat in name_lower for pat in SKIP_PATTERNS):
        print(f" Skipping '{filename}' (not an emotion dataset)")
        return True
    return False


def detect_text_col(df: pd.DataFrame):
    for col in df.columns:
        if any(h in col.lower() for h in TEXT_HINTS):
            return col
    best_col, best_avg = None, 0
    for col in df.columns:
        if df[col].dtype == object:
            avg = df[col].astype(str).str.len().mean()
            if avg > best_avg:
                best_avg, best_col = avg, col
    return best_col if best_avg > 15 else None


def detect_emotion_col(df: pd.DataFrame, text_col):
    for col in df.columns:
        if col == text_col:
            continue
        if any(h in col.lower() for h in EMOTION_HINTS):
            return col
    for col in df.columns:
        if col == text_col:
            continue
        if df[col].dtype == object:
            ratio = df[col].nunique() / max(len(df), 1)
            if ratio < 0.05:
                return col
    for col in df.columns:
        if col == text_col:
            continue
        if pd.api.types.is_integer_dtype(df[col]):
            if df[col].nunique() <= 30:
                return col
    return None


def load_file(filepath: str) -> pd.DataFrame | None:
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.tsv':
            return pd.read_csv(filepath, sep='\t', low_memory=False)
        elif ext in ('.xlsx', '.xls'):
            return pd.read_excel(filepath)
        else:
            return pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f" X Could not load: {e}")
        return None


print("=" * 70)
print("AUTO DATASET PREPARATION")
print("=" * 70)
print(f"\n Scanning {RAW_DIR} for dataset files ...")

EXTENSIONS = {'.csv', '.xlsx', '.xls', '.tsv'}
raw_files = [
    f for f in os.listdir(RAW_DIR)
    if os.path.splitext(f)[1].lower() in EXTENSIONS
    and not should_skip(f)
]

if raw_files:
    print(f"√ Found {len(raw_files)} usable file(s):")
    for f in raw_files:
        size_mb = os.path.getsize(os.path.join(RAW_DIR, f)) / 1_048_576
        print(f" • {f} ({size_mb:.1f} MB)")
else:
    print(" No usable CSV/Excel files found in data/raw/")


all_data = []

for filename in raw_files:
    filepath = os.path.join(RAW_DIR, filename)
    print(f"\n Processing: {filename}")

    df = load_file(filepath)
    if df is None:
        continue

    print(f" Rows: {len(df):,} | Columns: {df.columns.tolist()}")

    text_col = detect_text_col(df)
    emotion_col = detect_emotion_col(df, text_col)

    if text_col and emotion_col:
        print(f" √ text -> '{text_col}' | label -> '{emotion_col}'")

        chunk = pd.DataFrame({
            'text': df[text_col].astype(str),
            'emotion': df[emotion_col].astype(str).str.strip()
        }).dropna()

        chunk = chunk[chunk['text'].str.len() >= 10]
        all_data.append(chunk)
        print(f" √ Added {len(chunk):,} samples")
    else:
        print(f" Could not auto-detect columns - skipping")
        print(f" text_col={text_col}, emotion_col={emotion_col}")


print("\n" + "=" * 70)
print(" Downloading online datasets ...")

try:
    from datasets import load_dataset

    print("\n [1/2] dair-ai/emotion (~436K samples) ...")
    try:
        ds = load_dataset("dair-ai/emotion", trust_remote_code=True)
        EMOTION_MAP = {0: 'Sadness', 1: 'Joy', 2: 'Love',
                       3: 'Anger', 4: 'Fear', 5: 'Surprise'}
        parts = [pd.DataFrame(ds[split]) for split in ds.keys()]
        df_em = pd.concat(parts, ignore_index=True)
        df_em['emotion'] = df_em['label'].map(EMOTION_MAP)
        df_em = df_em[['text', 'emotion']].dropna()
        all_data.append(df_em)
        print(f" √ Added {len(df_em):,} samples from dair-ai/emotion")
    except Exception as e:
        print(f" dair-ai/emotion failed: {e}")

    print("\n [2/2] google-research-datasets/go_emotions (~58K samples) ...")
    try:
        GO_LABELS = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

        def _first_label(label_list):
            for i, v in enumerate(label_list):
                if v == 1:
                    return GO_LABELS[i].capitalize()
            return 'Neutral'

        go_ds = load_dataset(
            "google-research-datasets/go_emotions", "simplified",
            trust_remote_code=True
        )
        go_parts = [pd.DataFrame(go_ds[split]) for split in go_ds.keys()]
        df_go = pd.concat(go_parts, ignore_index=True)
        df_go['emotion'] = df_go['labels'].apply(_first_label)
        df_go = df_go[['text', 'emotion']].dropna()
        all_data.append(df_go)
        print(f" √ Added {len(df_go):,} samples from GoEmotions")
    except Exception as e:
        print(f" GoEmotions failed: {e}")

except ImportError:
    print(" `datasets` library not installed - skipping online download")
    print(" (It will be installed by run.bat before this step)")


def clean_and_normalize_emotions(df):
    df['emotion'] = df['emotion'].astype(str).str.strip().str.lower()

    mapping = {
        'neutral': 'Neutral',
        'normal': 'Neutral',
        'non-suicide': 'Neutral',
        'no_suicida': 'Neutral',
        'non-suicidal': 'Neutral',
        'no-suicide': 'Neutral',

        'joy': 'Joy',
        'joyful': 'Joy',
        'happiness': 'Joy',
        'happy': 'Joy',
        'fun': 'Joy',
        'excited': 'Joy',
        'excitement': 'Joy',
        'amusement': 'Joy',
        'pleasure': 'Joy',
        'pride': 'Joy',
        'optimism': 'Joy',

        'sadness': 'Sadness',
        'sad': 'Sadness',
        'grief': 'Sadness',
        'sorrow': 'Sadness',
        'depressed': 'Sadness',
        'depression': 'Sadness',
        'worry': 'Sadness',
        'disappointment': 'Sadness',
        'remorse': 'Sadness',
        'boredom': 'Sadness',

        'anger': 'Anger',
        'angry': 'Anger',
        'hate': 'Anger',
        'hateful': 'Anger',
        'irritated': 'Anger',
        'annoyance': 'Anger',
        'annoyed': 'Anger',
        'fury': 'Anger',
        'disgust': 'Anger',
        'disapproval': 'Anger',

        'fear': 'Fear',
        'afraid': 'Fear',
        'scared': 'Fear',
        'anxiety': 'Fear',
        'anxious': 'Fear',
        'panic': 'Fear',
        'nervousness': 'Fear',
        'nervous': 'Fear',

        'love': 'Love',
        'affection': 'Love',
        'caring': 'Love',
        'desire': 'Love',
        'passion': 'Love',
        'gratitude': 'Love',
        'admiration': 'Love',

        'surprise': 'Surprise',
        'surprised': 'Surprise',
        'shocked': 'Surprise',
        'amazement': 'Surprise',

        'suicide': 'Suicidal',
        'suicidal': 'Suicidal',
        'suicida': 'Suicidal',
    }

    df['emotion'] = df['emotion'].map(mapping)
    df = df.dropna(subset=['emotion'])
    return df


if not all_data:
    print("\nX No data loaded! Drop at least one CSV into data/raw/ and retry.")
    raise SystemExit(1)

print("\n" + "=" * 70)
print(" Combining all datasets ...")

df_all = pd.concat(all_data, ignore_index=True)

df_all = clean_and_normalize_emotions(df_all)
df_all['text'] = df_all['text'].astype(str).str.strip()
df_all = df_all[df_all['text'].str.len() >= 10]

counts = df_all['emotion'].value_counts()
valid = counts[counts >= 20].index
df_all = df_all[df_all['emotion'].isin(valid)]

print(f"\n√ Combined dataset:")
print(f" Total samples : {len(df_all):,}")
print(f" Unique emotions: {df_all['emotion'].nunique()}")

print("\n Top Emotions (Cleaned):")
for emo, cnt in df_all['emotion'].value_counts().items():
    bar = "█" * min(40, int(cnt / max(counts) * 40))
    print(f" {emo:<22} {cnt:>7,} {bar}")

df_all.to_csv(COMBINED_OUT, index=False)
print(f"\n Saved combined -> {COMBINED_OUT}")

print("\n Creating balanced version (max 10000 samples/emotion) ...")
balanced = []
for emo in df_all['emotion'].unique():
    chunk = df_all[df_all['emotion'] == emo]
    n = min(10000, len(chunk))
    balanced.append(chunk.sample(n=n, random_state=42))

df_bal = pd.concat(balanced, ignore_index=True).sample(frac=1, random_state=42)
df_bal.to_csv(BALANCED_OUT, index=False)
print(f"√ Balanced: {len(df_bal):,} samples across {df_bal['emotion'].nunique()} emotions")
print(f" Saved balanced -> {BALANCED_OUT}")

print("\n" + "=" * 70)
print(" Dataset preparation complete!")
print(f" Next step: venv\\Scripts\\python.exe scripts\\train_auto_model.py")
print("=" * 70)
