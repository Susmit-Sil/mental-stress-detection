import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from face_emotion_detector import analyze_face_emotion
from PIL import Image
import cv2


def evaluate_face_model():
    """
    Evaluates FER + DeepFace models on FER2013 Test dataset
    """
    
    # ⬇️ CHANGE THIS to your actual FER folder path
    test_folder = 'FER/Test'  # or 'E:/Project/Mental-Stress-Detection/FER/Test'
    
    if not os.path.exists(test_folder):
        print(f"❌ Test folder not found: {test_folder}")
        print("   Update the path in evaluate_face_model.py")
        return
    
    # Emotion mapping (FER uses these 7 emotions)
    emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    print("🔍 Scanning test images...")
    
    true_labels = []
    predicted_labels = []
    total_images = 0
    processed = 0
    
    # Count total images first
    for emotion in emotion_folders:
        emotion_path = os.path.join(test_folder, emotion)
        if os.path.exists(emotion_path):
            total_images += len([f for f in os.listdir(emotion_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"📊 Found {total_images:,} test images across {len(emotion_folders)} emotions\n")
    
    # Process each emotion folder
    for emotion in emotion_folders:
        emotion_path = os.path.join(test_folder, emotion)
        
        if not os.path.exists(emotion_path):
            print(f"⚠️  Skipping {emotion} (folder not found)")
            continue
        
        image_files = [f for f in os.listdir(emotion_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"🔄 Processing {emotion}: {len(image_files)} images...")
        
        for img_file in image_files:
            img_path = os.path.join(emotion_path, img_file)
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                # Analyze emotion
                result = analyze_face_emotion(pil_img)
                
                if result['status'] == 'success':
                    predicted_emotion = result['emotion'].lower()
                    true_labels.append(emotion)
                    predicted_labels.append(predicted_emotion)
                    processed += 1
                    
                    # Progress indicator
                    if processed % 100 == 0:
                        print(f"   ✅ Processed {processed}/{total_images}")
                
            except Exception as e:
                continue
    
    print(f"\n✅ Successfully processed {processed}/{total_images} images\n")
    
    if len(true_labels) == 0:
        print("❌ No images were successfully processed!")
        return
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, 
        average='weighted', 
        zero_division=0,
        labels=emotion_folders
    )
    
    # Print results
    print(f"{'='*60}")
    print(f"✅ FACE MODEL (FER + DeepFace) METRICS:")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"{'='*60}")
    
    # Per-emotion accuracy
    print(f"\n📊 PER-EMOTION ACCURACY:")
    print(f"{'='*60}")
    cm = confusion_matrix(true_labels, predicted_labels, labels=emotion_folders)
    for i, emotion in enumerate(emotion_folders):
        emotion_accuracy = cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"{emotion.capitalize():10s}: {emotion_accuracy*100:5.2f}% ({cm[i][i]}/{cm[i].sum()})")
    print(f"{'='*60}")
    
    # Save metrics to JSON
    face_metrics = {
        'model': 'FER + DeepFace',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'total_images_tested': processed,
        'num_classes': len(emotion_folders),
        'emotions': emotion_folders,
        'per_emotion_accuracy': {
            emotion: float(cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0)
            for i, emotion in enumerate(emotion_folders)
        }
    }
    
    with open('face_model_metrics.json', 'w') as f:
        json.dump(face_metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to: face_model_metrics.json")


if __name__ == '__main__':
    print("🚀 Starting Face Model Evaluation...")
    print("=" * 60)
    evaluate_face_model()
    print("\n✅ Evaluation complete!")



