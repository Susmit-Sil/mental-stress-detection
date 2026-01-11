import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os

def train_model():
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*70)
    print("TRAINING MEGA EMOTION MODEL - MENTAL STRESS DETECTION")
    print("="*70)
    print(f"\n🚀 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # ========== CHOOSE WHICH DATASET TO USE ==========
    print("\n📁 Available datasets:")
    
    mega_exists = os.path.exists('mega_emotion_dataset.csv')
    balanced_exists = os.path.exists('balanced_emotion_dataset.csv')
    
    if mega_exists:
        mega_size = len(pd.read_csv('mega_emotion_dataset.csv'))
        print(f"   1. mega_emotion_dataset.csv ({mega_size:,} samples, 2-3 hour training)")
    else:
        print(f"   1. mega_emotion_dataset.csv (NOT FOUND)")
        
    if balanced_exists:
        balanced_size = len(pd.read_csv('balanced_emotion_dataset.csv'))
        print(f"   2. balanced_emotion_dataset.csv ({balanced_size:,} samples, 30-45 min training)")
    else:
        print(f"   2. balanced_emotion_dataset.csv (NOT FOUND)")
    
    if not mega_exists and not balanced_exists:
        print("\n❌ ERROR: No dataset found!")
        print("   Please run: python prepare_mega_dataset.py first")
        return
    
    # Choose dataset (use balanced by default for speed)
    dataset_file = 'balanced_emotion_dataset.csv' if balanced_exists else 'mega_emotion_dataset.csv'
    
    # OPTION: Uncomment line below to use FULL mega dataset (500K+ samples)
    # dataset_file = 'mega_emotion_dataset.csv'
    
    print(f"\n✅ Using: {dataset_file}")

    # Load dataset
    print(f"\n📁 Loading dataset...")
    df = pd.read_csv(dataset_file)
    
    print(f"\n📊 Dataset Information:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check if columns exist
    if 'text' not in df.columns or 'emotion' not in df.columns:
        print(f"\n❌ ERROR: Dataset doesn't have 'text' and 'emotion' columns!")
        print(f"   Found columns: {df.columns.tolist()}")
        print(f"   Please run: python prepare_mega_dataset.py")
        return
    
    texts = df['text'].astype(str).tolist()
    labels = df['emotion'].astype(str).tolist()

    print(f"   Unique emotions: {len(set(labels))}")

    # Encode labels
    print("\n🔢 Encoding emotions...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    print(f"✅ Encoded {len(label_encoder.classes_)} emotion classes")
    print(f"   Sample emotions: {label_encoder.classes_[:15].tolist()}")

    # Split data
    print("\n✂️  Splitting dataset...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.1, random_state=42
    )
    print(f"✅ Train: {len(train_texts):,}, Validation: {len(val_texts):,}")

    # Load DistilBERT
    print("\n🤖 Loading DistilBERT model...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label_encoder.classes_)
    ).to(device)
    print(f"✅ Model loaded with {len(label_encoder.classes_)} output classes")

    # Tokenize
    print("\n🔤 Tokenizing texts...")
    print(f"   This will take ~{len(texts) // 10000} minutes...")
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding=True, 
        max_length=128,
        return_tensors='pt'
    )
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding=True, 
        max_length=128,
        return_tensors='pt'
    )
    print("✅ Tokenization complete!")

    # Create PyTorch Dataset
    class EmotionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = EmotionDataset(train_encodings, train_labels)
    val_dataset = EmotionDataset(val_encodings, val_labels)

    # Training configuration
    print("\n⚙️  Setting up training...")
    training_args = TrainingArguments(
        output_dir='./results_mega',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs_mega',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        dataloader_num_workers=0,  # Set to 0 for Windows
        report_to="none",
    )

    # Metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n🏋️  Training on GPU...")
    estimated_time = "30-45 minutes" if len(texts) < 100000 else "2-3 hours"
    print(f"   Estimated time: {estimated_time}")
    print(f"   Training on {len(train_texts):,} samples")
    print("="*70)

    trainer.train()

    # Evaluate
    print("\n📊 Final Evaluation...")
    results = trainer.evaluate()
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"   Final Validation Accuracy: {results['eval_accuracy']*100:.2f}%")

    # Save model
    print("\n💾 Saving model...")
    model.save_pretrained('./emotion_model_mega')
    tokenizer.save_pretrained('./emotion_model_mega')
    pickle.dump(label_encoder, open('label_encoder_mega.pkl', 'wb'))

    # Save metadata
    metadata = {
        'total_samples': len(texts),
        'num_classes': len(label_encoder.classes_),
        'accuracy': results['eval_accuracy'],
        'emotions': label_encoder.classes_.tolist(),
        'dataset_used': dataset_file
    }
    import json
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ MODEL SAVED SUCCESSFULLY!")
    print("   - Model: ./emotion_model_mega/")
    print("   - Label encoder: label_encoder_mega.pkl")
    print("   - Metadata: model_metadata.json")
    print(f"\n🎉 Ready to launch chatbot with {len(texts):,} training samples!")

# THIS IS REQUIRED FOR WINDOWS!
if __name__ == '__main__':
    train_model()
