import os
import sys
import json
import time
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# ----- CONFIG -----
BASE_DIR       = Path(__file__).resolve().parent.parent
FER_TRAIN_DIR  = BASE_DIR / "FER" / "Train"
FER_TEST_DIR   = BASE_DIR / "FER" / "Test"
FRIENDS_DIR    = BASE_DIR / "datasets" / "friends_cropped"
MODEL_OUT      = BASE_DIR / "models" / "custom_face_model.pth"
LABELS_OUT     = BASE_DIR / "models" / "custom_face_labels.json"

IMG_SIZE       = 48
BATCH_SIZE     = 64
EPOCHS         = 20
LR             = 1e-3
FRIENDS_WEIGHT = 5    # Each friend crop counts as N FER images (boosts friend accuracy)
if not torch.cuda.is_available():
    raise RuntimeError("GPU (CUDA) is not available! Training on CPU is disabled to prevent slow performance. Please ensure CUDA drivers and GPU-enabled PyTorch are installed.")
DEVICE         = torch.device("cuda")

# All emotions (only ones with data will be used)
ALL_EMOTIONS   = ["angry", "happy", "neutral", "sad", "surprise", "fear", "disgust"]


# ==================== DATASET ====================

class EmotionDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        """samples: list of (image_path, label_index)"""
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")       # Grayscale (48x48)
        img = img.convert("RGB")                  # MobileNet expects 3 channels
        if self.transform:
            img = self.transform(img)
        return img, label


def collect_samples(root: Path, label_map: dict, weight: int = 1) -> list:
    """Walks root/{emotion}/ folders, returns [(path, label)] * weight."""
    samples = []
    if not root.exists():
        return samples

    for emotion in ALL_EMOTIONS:
        folder = root / emotion
        if not folder.exists() or emotion not in label_map:
            continue
        label = label_map[emotion]
        files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        for f in files:
            samples.extend([(str(f), label)] * weight)  # repeat = higher weight

    return samples


def build_label_map(fer_dir: Path, friends_dir: Path) -> dict:
    """Only include emotions that have at least 1 image in FER or friends."""
    emotions_with_data = set()

    for emotion in ALL_EMOTIONS:
        fer_folder     = fer_dir / emotion
        friends_folder = friends_dir / emotion
        if (fer_folder.exists() and any(fer_folder.glob("*.jpg"))) or \
           (friends_folder.exists() and any(friends_folder.glob("*.jpg"))):
            emotions_with_data.add(emotion)

    # Stable sort to keep label indices consistent across runs
    sorted_emotions = sorted(emotions_with_data)
    return {e: i for i, e in enumerate(sorted_emotions)}


# ==================== MODEL ====================

def build_model(num_classes: int) -> nn.Module:
    """MobileNetV3-Small - fast, lightweight, great for real-time."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Replace final classifier to match our emotion count
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model.to(DEVICE)


# ==================== TRAINING ====================

def make_weighted_sampler(samples: list, num_classes: int):
    """Balance classes automatically so no emotion dominates training."""
    labels = [s[1] for s in samples]
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    class_counts = np.maximum(class_counts, 1)   # avoid division by zero
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("  STEP 2 — Train Custom Face Emotion Model")
    print("=" * 60)
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  FER dir : {FER_TRAIN_DIR.resolve()}")
    print(f"  Friends : {FRIENDS_DIR.resolve()}")
    print("=" * 60)

    # Build label map (only emotions with actual data)
    label_map = build_label_map(FER_TRAIN_DIR, FRIENDS_DIR)
    if not label_map:
        print("[X] No emotion data found! Check FER/Train/ and datasets/friends_cropped/")
        return

    idx_to_emotion = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    print(f"\nActive emotions ({num_classes}): {list(label_map.keys())}")

    # Collect samples
    train_samples = collect_samples(FER_TRAIN_DIR, label_map, weight=1)
    val_samples   = collect_samples(FER_TEST_DIR,  label_map, weight=1)
    friend_train  = collect_samples(FRIENDS_DIR,   label_map, weight=FRIENDS_WEIGHT)
    train_samples.extend(friend_train)

    print("\nDataset breakdown:")
    print(f"   FER Train     : {len(train_samples) - len(friend_train):,} samples")
    print(f"   Friends (x{FRIENDS_WEIGHT})  : {len(friend_train):,} entries")
    print(f"   Total Train   : {len(train_samples):,}")
    print(f"   Val           : {len(val_samples):,}")

    if not train_samples:
        print("[X] No training samples found!")
        return

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # DataLoaders
    sampler      = make_weighted_sampler(train_samples, num_classes)
    train_loader = DataLoader(EmotionDataset(train_samples, train_tf),
                              batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=DEVICE.type == "cuda")
    val_loader   = DataLoader(EmotionDataset(val_samples, val_tf),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=DEVICE.type == "cuda")

    # Model, loss, optimizer, scheduler
    model     = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...\n")
    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        marker = " *" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch:2d}/{EPOCHS}  "
              f"| Train {tr_acc*100:5.2f}%  "
              f"| Val {val_acc*100:5.2f}%  "
              f"| {elapsed:.1f}s{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), MODEL_OUT)

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"  BEST MODEL: Epoch {best_epoch} -> Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"{'='*60}")

    # Load best weights for final report
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    _, final_acc, final_preds, final_labels = eval_epoch(model, val_loader, criterion)

    emotion_names = [idx_to_emotion[i] for i in range(num_classes)]
    print("\nPer-Emotion Report:")
    print(classification_report(final_labels, final_preds,
                                target_names=emotion_names, zero_division=0))

    # Save label map
    with open(LABELS_OUT, "w") as f:
        json.dump({"label_map": label_map, "idx_to_emotion": idx_to_emotion,
                   "num_classes": num_classes, "img_size": IMG_SIZE}, f, indent=2)

    print(f"\n[√] Model saved  : {MODEL_OUT}")
    print(f"[√] Labels saved : {LABELS_OUT}")
    print("\nCustom model is now active in the live app!")
    print("   Run: run.bat   (or venv\\Scripts\\streamlit.exe run chatbot_mega.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
