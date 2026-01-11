import shutil
import os

print("🧹 Cleaning up old model files...")
print("="*60)

# List of old files/folders to remove
old_files = [
    'combined_dataset.csv',
    'emotion_model_combined/',
    'label_encoder_combined.pkl',
    'emotion_model_final/',
    'label_encoder_final.pkl',
    'results/',
    'logs/',
    'chatbot_combined.py',  # Old chatbot file
]

deleted = []
skipped = []

for item in old_files:
    if os.path.exists(item):
        try:
            if os.path.isfile(item):
                os.remove(item)
                deleted.append(item)
                print(f"✅ Deleted file: {item}")
            elif os.path.isdir(item):
                shutil.rmtree(item)
                deleted.append(item)
                print(f"✅ Deleted folder: {item}")
        except Exception as e:
            print(f"⚠️  Could not delete {item}: {e}")
            skipped.append(item)
    else:
        skipped.append(item)

print("\n" + "="*60)
print(f"✅ Cleanup complete!")
print(f"   Deleted: {len(deleted)} items")
print(f"   Skipped: {len(skipped)} items (not found)")
print("\n💡 You can now train the mega model without conflicts!")
