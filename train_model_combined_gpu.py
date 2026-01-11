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

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print("MENTAL STRESS DETECTION - GPU TRAINING")
print("="*60)
print(f"\n🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Load ONLY sentiment dataset (better for mental health!)
print("\n📁 Loading sentiment dataset...")
df = pd.read_csv('sentimentdataset.csv')
df['Text'] = df['Text'].astype(str)
df['Sentiment'] = df['Sentiment'].str.strip()

# Filter rare emotions
emotion_counts = df['Sentiment'].value_counts()
valid_emotions = emotion_counts[emotion_counts >= 2].index
df = df[df['Sentiment'].isin(valid_emotions)].copy()

texts = df['Text'].tolist()
labels = df['Sentiment'].tolist()

print(f"✅ Dataset: {len(texts)} samples")
print(f"   Unique emotions: {len(set(labels))}")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42
)
print(f"✅ Train: {len(train_texts)}, Validation: {len(val_texts)}")

# Load model
print("\n🤖 Loading DistilBERT...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(label_encoder.classes_)
).to(device)

# Tokenize
print("\n🔤 Tokenizing...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Dataset class
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)

# Training config
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # More epochs for small dataset
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# Trainer
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
print("\n🏋️  Training...")
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"\n✅ Validation Accuracy: {results['eval_accuracy']*100:.2f}%")

# Save
model.save_pretrained('./emotion_model_final')
tokenizer.save_pretrained('./emotion_model_final')
pickle.dump(label_encoder, open('label_encoder_final.pkl', 'wb'))

print("\n✅ MODEL SAVED!")
