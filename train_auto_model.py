import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pickle
import os
import json  # ⬅️ ADDED


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
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
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
    
    # ⬇️ ADDED: Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    # ⬆️ END ADDED
    
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
        compute_metrics=compute_metrics  # ⬅️ CHANGED from lambda
    )
    
    print("🏋️  Training...")
    trainer.train()
    
    results = trainer.evaluate()
    
    # ⬇️ CHANGED: Print all 4 metrics
    print(f"\n{'='*60}")
    print(f"✅ TEXT MODEL (BERT) METRICS:")
    print(f"{'='*60}")
    print(f"Accuracy:  {results['eval_accuracy']*100:.2f}%")
    print(f"Precision: {results['eval_precision']*100:.2f}%")
    print(f"Recall:    {results['eval_recall']*100:.2f}%")
    print(f"F1-Score:  {results['eval_f1']*100:.2f}%")
    print(f"{'='*60}")
    # ⬆️ END CHANGED
    
    print("\n💾 Saving...")
    model.save_pretrained('./emotion_model_auto')
    tokenizer.save_pretrained('./emotion_model_auto')
    pickle.dump(label_encoder, open('label_encoder_auto.pkl', 'wb'))
    
    # ⬇️ ADDED: Save metrics to JSON
    text_metrics = {
        'model': 'BERT',
        'accuracy': float(results['eval_accuracy']),
        'precision': float(results['eval_precision']),
        'recall': float(results['eval_recall']),
        'f1': float(results['eval_f1']),
        'total_samples': len(df),
        'num_classes': len(label_encoder.classes_)
    }
    with open('text_model_metrics.json', 'w') as f:
        json.dump(text_metrics, f, indent=2)
    # ⬆️ END ADDED
    
    print("✅ Done!")
    print("✅ Metrics saved to: text_model_metrics.json")  # ⬅️ ADDED


if __name__ == '__main__':
    train()
