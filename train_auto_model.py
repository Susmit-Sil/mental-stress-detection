import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Training on:", device)
    
    # Use auto-generated dataset
    dataset_file = 'auto_balanced_dataset.csv'
    
    if not os.path.exists(dataset_file):
        print(f"❌ {dataset_file} not found!")
        print("   Run: python prepare_auto_dataset.py first")
        return
    
    df = pd.read_csv(dataset_file)
    print(f"✅ Loaded {len(df):,} samples")
    
    texts = df['text'].tolist()
    labels = df['emotion'].tolist()
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.1, random_state=42
    )
    
    print(f"✅ Train: {len(train_texts):,}, Val: {len(val_texts):,}")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=len(label_encoder.classes_)
    ).to(device)
    
    print("🔤 Tokenizing...")
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, enc, labels):
            self.enc = enc
            self.labels = labels
        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item['labels'] = torch.tensor(self.labels[i])
            return item
        def __len__(self):
            return len(self.labels)
    
    train_dataset = Dataset(train_enc, train_labels)
    val_dataset = Dataset(val_enc, val_labels)
    
    training_args = TrainingArguments(
        output_dir='./results_auto',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda x: {'accuracy': accuracy_score(x[1], np.argmax(x[0], axis=1))}
    )
    
    print("🏋️  Training...")
    trainer.train()
    
    results = trainer.evaluate()
    print(f"\n✅ Accuracy: {results['eval_accuracy']*100:.2f}%")
    
    print("💾 Saving...")
    model.save_pretrained('./emotion_model_auto')
    tokenizer.save_pretrained('./emotion_model_auto')
    pickle.dump(label_encoder, open('label_encoder_auto.pkl', 'wb'))
    
    print("✅ Done!")

if __name__ == '__main__':
    train()
