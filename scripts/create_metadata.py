import json
import os
import pandas as pd

print("Creating model metadata...")

# Check your actual data
metadata = {
    "model_name": "emotion_model_mega",  # or emotion_model_best
    "model_type": "distilbert-base-uncased",  # or roberta-base if you upgraded
    "total_samples": 21575,  # UPDATE THIS
    "num_classes": 25,       # UPDATE THIS
    "accuracy": 0.851,       # UPDATE THIS (as decimal, e.g., 0.95 for 95%)
    "precision": 0.85,
    "recall": 0.84,
    "f1": 0.85,
    "training_date": "2026-01-12",
    "datasets_used": [
        "sentimentdataset.csv",
        "Combined Data.csv",
        "dair-ai/emotion"
    ]
}

# Try to read actual dataset size
try:
    if os.path.exists("BEST_COMBINED_DATASET.csv"):
        df = pd.read_csv("BEST_COMBINED_DATASET.csv")
        metadata['total_samples'] = len(df)
        metadata['num_classes'] = df['emotion'].nunique()
        print(f"✅ Read from BEST_COMBINED_DATASET.csv")
    elif os.path.exists("ALL_TEXT_DATA_FOR_COMPARISON.csv"):
        df = pd.read_csv("ALL_TEXT_DATA_FOR_COMPARISON.csv")
        metadata['total_samples'] = len(df)
        metadata['num_classes'] = df['emotion'].nunique()
        print(f"✅ Read from ALL_TEXT_DATA_FOR_COMPARISON.csv")
except:
    print("⚠️ Could not read dataset, using default values")

# Save metadata
with open('model_metadata_mega.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Created model_metadata_mega.json")
print(f"   Samples: {metadata['total_samples']:,}")
print(f"   Classes: {metadata['num_classes']}")
print(f"   Accuracy: {metadata['accuracy']*100:.1f}%")
