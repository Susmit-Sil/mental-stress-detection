import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pickle
import os
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FILE = os.path.join(BASE_DIR, 'data', 'auto_balanced_dataset.csv')
MODEL_OUT = os.path.join(BASE_DIR, 'models', 'emotion_model_auto')
ENCODER_OUT = os.path.join(BASE_DIR, 'models', 'label_encoder_auto.pkl')
METRICS_OUT = os.path.join(BASE_DIR, 'models', 'text_model_metrics.json')


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = labels

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item

    def __len__(self):
        return len(self.labels)


def train():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU (CUDA) is not available! Training on CPU is disabled to prevent slow performance. Please ensure CUDA drivers and GPU-enabled PyTorch are installed.")
    device = torch.device('cuda')
    print(f" Training on GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(DATASET_FILE):
        print(f"X {DATASET_FILE} not found!")
        print(" Run: python prepare_auto_dataset.py first")
        return

    df = pd.read_csv(DATASET_FILE)
    print(f"√ Loaded {len(df):,} samples | {df['emotion'].nunique()} emotions")

    texts = df['text'].tolist()
    labels = df['emotion'].tolist()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.1, random_state=42
    )
    print(f"√ Train: {len(train_texts):,} | Val: {len(val_texts):,}")

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=len(label_encoder.classes_)
    ).to(device)

    print(" Tokenizing (max_length=256 to capture long Reddit posts) ...")
    train_enc = tokenizer(
        train_texts, truncation=True, padding=True,
        max_length=256, return_tensors='pt'
    )
    val_enc = tokenizer(
        val_texts, truncation=True, padding=True,
        max_length=256, return_tensors='pt'
    )

    train_ds = EmotionDataset(train_enc, train_labels)
    val_ds = EmotionDataset(val_enc, val_labels)

    training_args = TrainingArguments(
        output_dir='./results_auto',
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=32,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss', # Target lowest validation loss
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to='none',
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print(" Training ...")
    trainer.train()

    results = trainer.evaluate()

    print(f"\n{'='*60}")
    print("√ TEXT MODEL (RoBERTa) METRICS:")
    print(f"{'='*60}")
    print(f"Accuracy : {results['eval_accuracy']*100:.2f}%")
    print(f"Precision : {results['eval_precision']*100:.2f}%")
    print(f"Recall : {results['eval_recall']*100:.2f}%")
    print(f"F1-Score : {results['eval_f1']*100:.2f}%")
    print(f"{'='*60}")

    print("\n Saving model ...")
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    pickle.dump(label_encoder, open(ENCODER_OUT, 'wb'))

    metrics = {
        'model': 'RoBERTa',
        'accuracy': float(results['eval_accuracy']),
        'precision': float(results['eval_precision']),
        'recall': float(results['eval_recall']),
        'f1': float(results['eval_f1']),
        'total_samples': len(df),
        'num_classes': len(label_encoder.classes_),
        'classes': list(label_encoder.classes_),
    }
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"√ Model saved to : {MODEL_OUT}")
    print(f"√ Encoder saved to: {ENCODER_OUT}")
    print(f"√ Metrics saved to: {METRICS_OUT}")


if __name__ == '__main__':
    train()
