from fer.fer import FER
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import warnings
warnings.filterwarnings('ignore')

def preprocess_face_image(image):
    """
    Enhance image quality for better emotion detection
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Enhance brightness (helps with dark images)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    
    # Enhance contrast (makes facial features clearer)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    
    # Enhance sharpness (better feature detection)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.4)
    
    return image

def detect_emotion_from_image(image):
    """
    IMPROVED: Uses ensemble of FER + DeepFace for better accuracy
    """
    try:
        # Preprocess image first
        image = preprocess_face_image(image)
        img_array = np.array(image)
        
        # Store all model results
        all_predictions = []
        
        # ===== MODEL 1: FER with MTCNN (Better face detection) =====
        try:
            detector_fer = FER(mtcnn=True)  # MTCNN is better than Haar Cascade
            fer_results = detector_fer.detect_emotions(img_array)
            
            if fer_results and len(fer_results) > 0:
                fer_emotions = fer_results[0]['emotions']
                face_box = fer_results[0]['box']
                
                # Normalize to 0-100 scale
                fer_emotions_normalized = {k: v*100 for k, v in fer_emotions.items()}
                all_predictions.append({
                    'emotions': fer_emotions_normalized,
                    'weight': 0.6,  # FER is more reliable for emotions
                    'face_box': face_box
                })
                print("✅ FER detected face")
        except Exception as e:
            print(f"⚠️ FER failed: {e}")
        
        # ===== MODEL 2: DeepFace (Good for certain emotions) =====
        try:
            df_result = DeepFace.analyze(
                img_array,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(df_result, list):
                df_result = df_result[0]
            
            df_emotions = df_result['emotion']
            all_predictions.append({
                'emotions': df_emotions,
                'weight': 0.4,  # DeepFace as supporting model
                'face_box': None
            })
            print("✅ DeepFace detected emotions")
        except Exception as e:
            print(f"⚠️ DeepFace failed: {e}")
        
        # ===== ENSEMBLE: Weighted Average =====
        if not all_predictions:
            return {
                'success': False,
                'error': 'No face detected. Please ensure:\n- Face is clearly visible\n- Good lighting\n- Face is not too small'
            }
        
        # Combine predictions with weights
        emotion_keys = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        combined_emotions = {}
        
        for emotion in emotion_keys:
            weighted_sum = 0
            total_weight = 0
            
            for pred in all_predictions:
                if emotion in pred['emotions']:
                    weighted_sum += pred['emotions'][emotion] * pred['weight']
                    total_weight += pred['weight']
            
            if total_weight > 0:
                combined_emotions[emotion] = weighted_sum / total_weight
        
        # Get dominant emotion
        if not combined_emotions:
            return {'success': False, 'error': 'Could not analyze emotions'}
        
        dominant_emotion = max(combined_emotions, key=combined_emotions.get)
        confidence = combined_emotions[dominant_emotion]
        
        # Map to stress levels
        stress_mapping = {
            'angry': ('High Stress', '🔴', '#ff4444'),
            'disgust': ('High Stress', '🔴', '#ff4444'),
            'fear': ('High Stress', '🔴', '#ff4444'),
            'sad': ('High Stress', '🔴', '#ff4444'),
            'happy': ('Low Stress', '🟢', '#44ff44'),
            'surprise': ('Moderate Stress', '🟡', '#ffff44'),
            'neutral': ('Moderate Stress', '🟡', '#ffff44')
        }
        
        stress_level, stress_color, color_code = stress_mapping.get(
            dominant_emotion,
            ('Unknown', '⚪', '#cccccc')
        )
        
        # Get face box from first successful detection
        face_box = None
        for pred in all_predictions:
            if pred.get('face_box'):
                face_box = pred['face_box']
                break
        
        return {
            'success': True,
            'emotion': dominant_emotion.capitalize(),
            'confidence': confidence,
            'all_emotions': combined_emotions,
            'stress_level': stress_level,
            'stress_color': stress_color,
            'color_code': color_code,
            'face_box': face_box,
            'num_models': len(all_predictions)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error analyzing image: {str(e)}'
        }

def draw_emotion_on_image(image, result):
    """
    Draw enhanced bounding box and emotion labels on image
    """
    if not result['success']:
        return image
    
    img = np.array(image.copy())
    
    # Draw face bounding box if available
    if result.get('face_box'):
        x, y, w, h = result['face_box']
        
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Draw emotion label above face
        emotion_text = f"{result['emotion']}: {result['confidence']:.1f}%"
        
        # Background for text (makes it readable)
        (text_width, text_height), _ = cv2.getTextSize(
            emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(img, (x, y-40), (x+text_width+10, y), (0, 255, 0), -1)
        
        # Text
        cv2.putText(img, emotion_text, (x+5, y-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Draw stress level below face
        stress_text = f"{result['stress_color']} {result['stress_level']}"
        
        # Background for stress text
        (stress_width, stress_height), _ = cv2.getTextSize(
            stress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(img, (x, y+h), (x+stress_width+10, y+h+35), (0, 255, 0), -1)
        
        # Stress text
        cv2.putText(img, stress_text, (x+5, y+h+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        # No face box, just overlay text at top
        emotion_text = f"{result['emotion']}: {result['confidence']:.1f}%"
        stress_text = f"{result['stress_color']} {result['stress_level']}"
        
        cv2.putText(img, emotion_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(img, stress_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return Image.fromarray(img)

def analyze_face_emotion(image):
    """
    Wrapper function for evaluate_face_model.py compatibility
    Converts detect_emotion_from_image() output to expected format
    """
    result = detect_emotion_from_image(image)
    
    # Convert format
    if result.get('success'):
        return {
            'status': 'success',
            'emotion': result['emotion'].lower(),  # Convert to lowercase
            'confidence': result['confidence'] / 100.0,  # Convert to 0-1 scale
            'all_emotions': result['all_emotions'],
            'method': f"Ensemble ({result['num_models']} models)"
        }
    else:
        return {
            'status': 'error',
            'message': result.get('error', 'Unknown error'),
            'emotion': 'neutral',
            'confidence': 0.0
        }
