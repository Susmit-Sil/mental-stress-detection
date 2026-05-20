import json
import os
import pandas as pd
import pickle
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("="*70)
print("MODEL INFORMATION CHECKER")
print("="*70)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== CHECK 1: Model Metadata =====
print("\n📊 MODEL METADATA:")
metadata_files = [
    os.path.join(BASE_DIR, 'models', 'model_metadata.json'),
    os.path.join(BASE_DIR, 'models', 'model_metadata_mega.json'),
    os.path.join(BASE_DIR, 'models', 'model_metadata_best.json')
]

metadata_found = False
for mf in metadata_files:
    if os.path.exists(mf):
        with open(mf, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Found {os.path.basename(mf)}")
        print(f"\n   Total Training Samples: {metadata.get('total_samples', 'Unknown'):,}")
        print(f"   Number of Emotions: {metadata.get('num_classes', 'Unknown')}")
        print(f"   Model Accuracy: {metadata.get('accuracy', 0)*100:.2f}%")
        
        if 'dataset_used' in metadata:
            print(f"   Dataset File Used: {metadata['dataset_used']}")
        
        print(f"\n   Emotions Detected:")
        emotions = metadata.get('emotions', [])
        for i, emotion in enumerate(emotions[:20], 1):
            print(f"      {i}. {emotion}")
        if len(emotions) > 20:
            print(f"      ... and {len(emotions)-20} more")
        metadata_found = True
        break

if not metadata_found:
    print("⚠️  No model metadata files found in models/ folder")

# ===== CHECK 2: Available Dataset Files =====
print("\n" + "="*70)
print("📁 DATASET FILES IN FOLDER:")

dataset_files = [
    'sentimentdataset.csv',
    'Combined Data.csv',
    'Datafiniti_Hotel_Reviews.csv',
    'mega_emotion_dataset.csv',
    'balanced_emotion_dataset.csv',
    'auto_combined_dataset.csv',
    'auto_balanced_dataset.csv'
]

found_datasets = []
for file in dataset_files:
    path = os.path.join(BASE_DIR, 'data', file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n✅ {file}")
        print(f"   Rows: {len(df):,}")
        if 'emotion' in df.columns:
            print(f"   Unique emotions: {df['emotion'].nunique()}")
            print(f"   Top 5 emotions: {df['emotion'].value_counts().head(5).index.tolist()}")
        found_datasets.append((file, len(df)))

if not found_datasets:
    print("⚠️  No dataset files found in data/ folder")

# ===== CHECK 3: Label Encoder =====
print("\n" + "="*70)
print("🏷️  LABEL ENCODER:")

encoder_files = [
    'label_encoder_mega.pkl',
    'label_encoder_auto.pkl',
    'label_encoder_combined.pkl',
    'label_encoder_best.pkl'
]

for encoder_file in encoder_files:
    path = os.path.join(BASE_DIR, 'models', encoder_file)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            encoder = pickle.load(f)
        
        print(f"\n✅ {encoder_file}")
        print(f"   Classes: {len(encoder.classes_)}")
        print(f"   Sample classes: {encoder.classes_[:15].tolist()}")
        break

# ===== CHECK 4: Model Folder =====
print("\n" + "="*70)
print("🤖 MODEL FILES:")

model_folders = [
    'emotion_model_mega',
    'emotion_model_auto',
    'emotion_model_combined',
    'emotion_model_best'
]

for model_folder in model_folders:
    path = os.path.join(BASE_DIR, 'models', model_folder)
    if os.path.exists(path):
        print(f"\n✅ {model_folder}/")
        
        # Check size
        total_size = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        
        print(f"   Size: {total_size / (1024**2):.1f} MB")
        
        # Check config
        config_file = os.path.join(path, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"   Model type: {config.get('model_type', 'Unknown')}")
            print(f"   Vocab size: {config.get('vocab_size', 'Unknown'):,}")

# ===== SUMMARY =====
print("\n" + "="*70)
print("📋 TRAINING DATA SUMMARY:")
print("="*70)

if found_datasets:
    total_rows = sum(rows for _, rows in found_datasets)
    print(f"\nTotal rows across all datasets: {total_rows:,}")
    print(f"\nDatasets found:")
    for file, rows in found_datasets:
        percentage = (rows / total_rows) * 100 if total_rows > 0 else 0
        print(f"  • {file}: {rows:,} rows ({percentage:.1f}%)")

print("\n" + "="*70)
