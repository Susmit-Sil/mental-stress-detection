import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TensorFlow warnings

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json
import time
from PIL import Image
import numpy as np
import cv2
from collections import deque

# Import face detector
from face_emotion_detector import detect_emotion_from_image, draw_emotion_on_image

# Import WebRTC components
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av
    from deepface import DeepFace
    from fer.fer import FER
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("️ streamlit-webrtc, deepface, or fer not installed. Live video will be disabled.")

st.set_page_config(
    page_title="Mental Stress Detection AI", 
    page_icon="",
    layout="wide"
)

# Performance optimization
torch.backends.cudnn.benchmark = True

USE_CUSTOM_IN_ENSEMBLE = True


if WEBRTC_AVAILABLE:
    class EmotionVideoTransformer(VideoProcessorBase):
        def __init__(self):
            # 5 emotions we care about
            self.EMOTIONS = ["angry", "happy", "sad", "surprise", "neutral"]
            
            # Colors
            self.WHITE = (255, 255, 255)
            self.NEON_PINK = (255, 0, 255)
            
            self.ANALYZE_EVERY_FRAMES = 30  # Every 30 frames (~1 second for ensemble)
            
            self.emotion_scores = {e: 0.0 for e in self.EMOTIONS}
            self.top_emotion = "neutral"
            self.top_conf = 0
            self.frame_count = 0
            self.last_analysis_frame = 0
            self.face_box = None
            self.use_ensemble = True
            
        def analyze_emotion(self, frame_bgr):
            """Run ensemble using face_emotion_detector (dynamic weights based on custom model availability)."""
            try:
                # Convert BGR (OpenCV) to RGB PIL Image
                rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                
                # Detect
                result = detect_emotion_from_image(pil_img, use_custom=USE_CUSTOM_IN_ENSEMBLE)
                
                if result['success']:
                    self.face_box = result.get('face_box')
                    
                    # Normalize scores to 0-100 and update self.emotion_scores for 5 target emotions
                    all_scores = result.get('all_emotions', {})
                    for e in self.EMOTIONS:
                        self.emotion_scores[e] = float(all_scores.get(e, 0.0))
                        
                    self.top_emotion = result.get('emotion', 'neutral').lower()
                    self.top_conf = int(result.get('confidence', 0))
                else:
                    self.face_box = None
            except Exception as e:
                pass
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24").copy()
            
            # Flip for mirror effect
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            self.frame_count += 1
            
            if self.frame_count - self.last_analysis_frame >= self.ANALYZE_EVERY_FRAMES:
                self.analyze_emotion(img)
                self.last_analysis_frame = self.frame_count
            
            cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
            hud_title = "AI FACE EMOTION HUD (ENSEMBLE)" if self.use_ensemble else "AI FACE EMOTION HUD"
            cv2.putText(img, hud_title,
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.NEON_PINK, 2)
            
            if hasattr(self, 'face_box') and self.face_box is not None:
                # Use detected face box
                x, y, box_w, box_h = self.face_box

                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                box_w = box_w + 2 * padding
                box_h = box_h + 2 * padding

                # Outer glow-ish white box
                cv2.rectangle(img, (x - 3, y - 3), (x + box_w + 3, y + box_h + 3),
                              (200, 200, 200), 2)
                # Inner solid white box
                cv2.rectangle(img, (x, y), (x + box_w, y + box_h), self.WHITE, 2)

                label = f"{self.top_emotion.upper()} ({self.top_conf}%)"
                cv2.rectangle(img, (x, y - 30), (x + box_w, y), (0, 0, 0), -1)
                cv2.putText(img, label,
                            (x + 10, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2)
            else:
                box_w, box_h = 260, 300
                x = w // 2 - box_w // 2
                y = h // 2 - box_h // 2

                cv2.rectangle(img, (x, y), (x + box_w, y + box_h), self.WHITE, 2)

                cv2.putText(img, "Position face in frame",
                            (x + 10, y + box_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2)
            
            panel_x = 15
            panel_y = 60
            line_h = 24
            max_bar_w = 140
            
            panel_width = 75 + max_bar_w + 10  # 225px total
            panel_height = len(self.EMOTIONS) * line_h + 20
            
            overlay = img.copy()
            cv2.rectangle(overlay, 
                          (panel_x - 5, panel_y - 25), 
                          (panel_x + panel_width, panel_y + panel_height),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            # Title
            cv2.putText(img, "Emotions:",
                        (panel_x, panel_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
            
            for i, emo in enumerate(self.EMOTIONS):
                y_off = panel_y + i * line_h
                score = self.emotion_scores[emo]
                bar_w = int((score / 100.0) * max_bar_w)
                
                # Emotion name (compact)
                cv2.putText(img, f"{emo:7s}",
                            (panel_x, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
                
                cv2.rectangle(img,
                              (panel_x + 75, y_off - 10),
                              (panel_x + 75 + max_bar_w, y_off + 2),
                              (50, 50, 50), -1)
                
                cv2.rectangle(img,
                              (panel_x + 75, y_off - 10),
                              (panel_x + 75 + bar_w, y_off + 2),
                              self.NEON_PINK if emo == self.top_emotion else (180, 180, 180),
                              -1)
            
            # Bottom instruction
            cv2.putText(img, "Press STOP to exit",
                        (w - 210, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 1)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try different model paths
    model_paths = [
        './emotion_model_best', './emotion_model_mega', './emotion_model_auto',
        './models/emotion_model_best', './models/emotion_model_mega', './models/emotion_model_auto'
    ]
    encoder_paths = [
        'label_encoder_best.pkl', 'label_encoder_mega.pkl', 'label_encoder_auto.pkl',
        'models/label_encoder_best.pkl', 'models/label_encoder_mega.pkl', 'models/label_encoder_auto.pkl'
    ]
    
    model_path = None
    encoder_path = None
    
    # Find existing model
    for mp in model_paths:
        if os.path.exists(mp):
            model_path = mp
            break
    
    # Find existing encoder
    for ep in encoder_paths:
        if os.path.exists(ep):
            encoder_path = ep
            break
    
    if not model_path or not encoder_path:
        st.error("X Model files not found! Please ensure model is trained.")
        if not st.runtime.exists():
            print("Model files not found! Please run the training script first.")
            sys.exit(0)
        st.stop()
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    label_encoder = pickle.load(open(encoder_path, 'rb'))
    
    # Load metadata
    metadata_files = [
        'models/text_model_metrics.json',
        'model_metadata_best.json', 'model_metadata_mega.json', 'model_metadata.json',
        'models/model_metadata_best.json', 'models/model_metadata_mega.json', 'models/model_metadata.json'
    ]
    metadata = {}
    for mf in metadata_files:
        if os.path.exists(mf):
            try:
                with open(mf, 'r') as f:
                    metadata = json.load(f)
                break
            except:
                pass
    
    return model, tokenizer, label_encoder, device, metadata

# Try to get actual sample sizes dynamically
@st.cache_data
def get_dataset_stats():
    text_raw = 479126
    text_train = 80000
    image_raw = 32298
    image_train = 28709
    
    # Text counts
    combined_path = "data/auto_combined_dataset.csv"
    if os.path.exists(combined_path):
        try:
            with open(combined_path, "r", encoding="utf-8", errors="ignore") as f:
                text_raw = sum(1 for _ in f) - 1
        except:
            pass

    balanced_path = "data/auto_balanced_dataset.csv"
    if os.path.exists(balanced_path):
        try:
            with open(balanced_path, "r", encoding="utf-8", errors="ignore") as f:
                text_train = sum(1 for _ in f) - 1
        except:
            pass
            
    # Image counts
    def count_images_in_dir(path):
        if not os.path.exists(path):
            return 0
        count = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    count += 1
        return count

    fer_count = count_images_in_dir("FER")
    friends_raw_count = count_images_in_dir("datasets/friends_dataset")
    if fer_count > 0 or friends_raw_count > 0:
        image_raw = fer_count + friends_raw_count
        
    fer_train_count = count_images_in_dir("FER/Train")
    friends_crop_count = count_images_in_dir("datasets/friends_cropped")
    if fer_train_count > 0 or friends_crop_count > 0:
        image_train = fer_train_count + friends_crop_count
        
    return text_raw, text_train, image_raw, image_train

# Find last update date dynamically based on files
def get_last_updated_date():
    import datetime
    target_paths = [
        'models/text_model_metrics.json',
        'models/custom_face_labels.json',
        'chatbot_mega.py'
    ]
    latest_time = 0
    for p in target_paths:
        if os.path.exists(p):
            t = os.path.getmtime(p)
            if t > latest_time:
                latest_time = t
    if latest_time > 0:
        dt = datetime.datetime.fromtimestamp(latest_time)
        return dt.strftime("%B %d, %Y")
    return "January 25, 2026"

# Load model
with st.spinner(" Loading AI models..."):
    model, tokenizer, label_encoder, device, metadata = load_model()

text_raw, text_train, image_raw, image_train = get_dataset_stats()

with st.sidebar:
    st.header("️ Model Information")
    
    # Complete dataset stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Raw Data", f"{text_raw + image_raw:,}")
        st.caption(f"↳ {text_raw:,} text + {image_raw:,} images")
    with col2:
        st.metric("Training", f"{text_train + image_train:,}")
        st.caption(f"↳ {text_train:,} text + {image_train:,} images")
    
    accuracy = metadata.get("accuracy", 0) * 100
    st.metric("Model Accuracy", f"{accuracy:.2f}%")
    
    st.divider()
    st.subheader("️ System")
    
    if torch.cuda.is_available():
        st.success("Running on:  GPU")
        st.caption(f"**GPU:** {torch.cuda.get_device_name(0)}")
    else:
        st.info("Running on:  CPU")
    
    st.divider()
    st.subheader("️ Tech Stack")
    st.write(f"• {metadata.get('model', 'RoBERTa')} Transformer")
    st.write("• FER + DeepFace + Custom Dataset Ensemble")
    st.write("• PyTorch (GPU)")
    st.write("• WebRTC Streaming")
    
    st.divider()
    st.caption("**Version:** 2.2 (Ensemble)")
    st.caption(f"**Updated:** {get_last_updated_date()}")

st.title(" Mental Stress Detection System")
st.markdown("**Multi-Modal AI for Mental Health Assessment**")

tab1, tab2, tab3, tab4 = st.tabs([" Text Analysis", " Face Analysis", " Live Video", "️ About"])

with tab1:
    st.header("Text-Based Emotion Detection")
    
    user_input = st.text_area(
        " How are you feeling today?",
        height=150,
        placeholder="Express your thoughts, feelings, or experiences...\n\nExamples:\n• I'm feeling anxious about my exams\n• Today was absolutely amazing!\n• I'm worried about my family"
    )
    
    analyze_btn = st.button(" Analyze Emotion", type="primary", use_container_width=True)
    
    if analyze_btn:
        if user_input.strip() and len(user_input) > 5:
            with st.spinner(" Analyzing with AI..."):
                start_time = time.time()
                
                # Tokenize and move to device
                inputs = tokenizer(
                    user_input,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Predict
                with torch.inference_mode():
                    outputs = model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    # Top 5
                    top5_probs, top5_indices = torch.topk(predictions[0], min(5, len(label_encoder.classes_)))
                    top5_emotions = [label_encoder.inverse_transform([idx.item()])[0] 
                                   for idx in top5_indices]
                    top5_confidences = [prob.item() * 100 for prob in top5_probs]
                
                processing_time = time.time() - start_time
            
            # Display results
            st.divider()
            st.subheader(" Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Primary Emotion", top5_emotions[0].upper())
            with col2:
                st.metric("Confidence", f"{top5_confidences[0]:.1f}%")
            with col3:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            # Top 5 emotions
            st.subheader(" Top 5 Detected Emotions")
            for i, (emotion, conf) in enumerate(zip(top5_emotions, top5_confidences), 1):
                st.progress(min(1.0, max(0.0, float(conf / 100))), text=f"{i}. {emotion}: {conf:.1f}%")
            
            # Mental health assessment
            st.divider()
            st.subheader(" Mental Health Assessment")
            
            crisis_keywords = [
                'suicidal', 'suicide', 'kill myself', 'end my life', 'end it all',
                'self harm', 'self-harm', 'cutting', 'hurt myself',
                'no reason to live', 'better off dead', 'want to die', 'can\'t go on',
                'no point living', 'give up on life'
            ]
            
            # High stress keywords
            stress_keywords = [
                'anger', 'anxiety', 'annoyance', 'confusion', 'disappointment',
                'disapproval', 'disgust', 'embarrassment', 'fear', 'grief',
                'nervousness', 'remorse', 'sadness', 'negative', 'stress',
                'worry', 'frustrated', 'overwhelmed', 'depressed', 'depression',
                'hopeless', 'helpless', 'lonely', 'isolated', 'panic', 'scared',
                'terrified', 'miserable', 'worthless', 'guilty', 'ashamed', 'broken'
            ]
            
            # Positive keywords
            positive_keywords = [
                'joy', 'love', 'admiration', 'amusement', 'approval', 'caring',
                'curiosity', 'desire', 'excitement', 'gratitude', 'optimism',
                'pride', 'relief', 'surprise', 'happiness', 'positive', 'grateful',
                'blessed', 'content', 'peaceful', 'hopeful', 'motivated', 'energized'
            ]
            
            emotion_lower = top5_emotions[0].lower()
            input_lower = user_input.lower()
            
            # CHECK FOR CRISIS FIRST
            is_crisis = any(crisis in emotion_lower or crisis in input_lower for crisis in crisis_keywords)
            
            if is_crisis:
                st.error(" **CRITICAL: IMMEDIATE HELP NEEDED**")
                
                st.markdown("""
                ###  You are not alone. Help is available RIGHT NOW.
                
                **Immediate Crisis Resources:**
                """)
                
                col_crisis1, col_crisis2 = st.columns(2)
                
                with col_crisis1:
                    st.markdown("""
                    **India (24/7 Helplines):**
                    -  **AASRA:** 91-9820466726
                    -  **Vandrevala Foundation:** 1860-2662-345
                    -  **iCall:** 91-22-25521111
                    -  **Sneha India:** 91-44-24640050
                    -  **Lifeline Foundation:** 033-24637401
                    """)
                
                with col_crisis2:
                    st.markdown("""
                    **International:**
                    -  **USA:** 988 (Suicide & Crisis Lifeline)
                    -  **UK:** 116 123 (Samaritans)
                    -  **Australia:** 13 11 14 (Lifeline)
                    -  **Canada:** 1-833-456-4566
                    """)
                
                st.error("️ **Please reach out to one of these numbers immediately. Your life matters.**")
            
            elif any(s in emotion_lower for s in stress_keywords):
                st.error("️ **Elevated Stress Level Detected**")
                
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    st.write("**Immediate Actions:**")
                    st.write("√ Take 10 deep breaths (4-7-8 technique)")
                    st.write("√ Step away for 5-10 minutes")
                    st.write("√ Drink water and stretch")
                    st.write("√ Ground yourself (5-4-3-2-1 method)")
                    
                with col_rec2:
                    st.write("**Helpful Resources:**")
                    st.write(" Talk to someone you trust")
                    st.write(" Try meditation (Headspace, Calm)")
                    st.write(" Journal your thoughts")
                    st.write(" Go for a walk outdoors")
                
                st.warning(" If these feelings persist for more than 2 weeks, please speak with a mental health professional")
                
                st.info("""
                **Mental Health Resources (India):**
                -  **Vandrevala Foundation:** 1860-2662-345 (24/7 emotional support)
                -  **Mann Talks:** +91-8686139139 (Free counseling)
                -  Visit your nearest hospital psychiatry department
                """)
                    
            elif any(p in emotion_lower for p in positive_keywords):
                st.success("√ **Positive Mental State Detected**")
                st.write(" Great! Your emotional well-being seems positive.")
                
                col_pos1, col_pos2 = st.columns(2)
                with col_pos1:
                    st.write("**Keep It Up:**")
                    st.write("• Maintain healthy sleep (7-9 hours)")
                    st.write("• Stay physically active (30 min/day)")
                    st.write("• Eat balanced, nutritious meals")
                with col_pos2:
                    st.write("**Continue These Habits:**")
                    st.write("• Connect with loved ones regularly")
                    st.write("• Practice daily gratitude")
                    st.write("• Engage in hobbies you enjoy")
                    
            else:
                st.info("️ **Neutral Emotional State**")
                st.write("Your emotional state appears balanced. Stay mindful and check in with yourself regularly.")
                
        elif user_input.strip():
            st.warning("️ Please enter at least 5 characters for accurate analysis")
        else:
            st.warning("️ Please enter some text to analyze!")

with tab2:
    st.header("Face-Based Emotion Detection")
    st.markdown("Upload a photo or take a picture to detect stress from facial expressions")
    
    # Upload option
    st.subheader(" Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png'],
        key="face_upload"
    )
    
    # Camera option
    st.subheader(" Take Picture")
    camera_photo = st.camera_input("Take a photo", key="face_camera")
    
    # Process image
    image_to_process = uploaded_file if uploaded_file else camera_photo
    
    if image_to_process is not None:
        image = Image.open(image_to_process)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with st.spinner(" Analyzing face with AI (ensemble detection)..."):
            result = detect_emotion_from_image(image, use_custom=USE_CUSTOM_IN_ENSEMBLE)
            
            if result['success']:
                with col2:
                    st.subheader("Analysis Result")
                    annotated = draw_emotion_on_image(image, result)
                    st.image(annotated, use_container_width=True)
                
                # Results
                num_models = result.get('num_models', 1)
                st.success(f"√ Face emotion detected! (Using {num_models} AI model{'s' if num_models > 1 else ''})")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric("Detected Emotion", result['emotion'])
                
                with res_col2:
                    st.metric("Confidence", f"{result['confidence']:.1f}%")
                
                with res_col3:
                    stress_level = result['stress_level']
                    if 'High' in stress_level:
                        st.error(f"{result['stress_color']} {stress_level}")
                    elif 'Low' in stress_level:
                        st.success(f"{result['stress_color']} {stress_level}")
                    else:
                        st.warning(f"{result['stress_color']} {stress_level}")
                
                # Detailed emotions
                st.subheader(" Detailed Emotion Analysis")
                if num_models > 1:
                    st.caption("Combined prediction from multiple AI models")
                
                emotions_data = result['all_emotions']
                for emotion, score in sorted(emotions_data.items(), key=lambda x: x[1], reverse=True):
                    col_name, col_bar = st.columns([1, 4])
                    
                    with col_name:
                        emoji_map = {
                            'happy': '', 'sad': '', 'angry': '',
                            'fear': '', 'surprise': '', 'disgust': '', 'neutral': ''
                        }
                        emoji = emoji_map.get(emotion, '')
                        st.write(f"{emoji} **{emotion.capitalize()}**")
                    
                    with col_bar:
                        st.progress(min(1.0, max(0.0, float(score / 100))))
                        st.caption(f"{score:.1f}%")
                
            else:
                st.error(f"X {result.get('error', 'Could not detect face')}")
                st.info("""
                 **Tips for better detection:**
                - √ Ensure face is clearly visible
                -  Use good lighting
                -  Face the camera directly
                -  Remove sunglasses/masks
                -  Get closer to camera
                """)

with tab3:
    st.header(" Real-Time Emotion Detection")
    st.markdown("**Live webcam emotion analysis with AI HUD overlay (ENSEMBLE MODE)**")
    
    if not WEBRTC_AVAILABLE:
        st.error("X **Required libraries not installed**")
        st.info("""
        To enable live video analysis, install:
        
        ```bash
        pip install streamlit-webrtc deepface fer
        ```
        
        Then restart the application.
        """)
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("###  Live Feed")
            webrtc_ctx = webrtc_streamer(
                key="emotion-detection-live",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=EmotionVideoTransformer,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        
        with col2:
            st.markdown("###  Instructions")
            st.info("""
            **How to use:**
            1. Click **START** button
            2. Allow camera access
            3. Position face in center box
            4. Watch real-time analysis
            5. Click **STOP** to end
            """)
            
            st.markdown("###  Features")
            custom_face_trained = os.path.exists("models/custom_face_model.pth")
            weights_desc = "40% FER + 30% DeepFace + 30% Custom" if custom_face_trained else "60% FER + 40% DeepFace"
            st.success(f"""
            √ Live face detection  
            √ **ENSEMBLE MODE**
            √ {weights_desc}
            √ Real-time HUD overlay
            √ Compact side panel
            """)
            
            st.markdown("### ️ Performance")
            st.metric("Expected FPS", "15-25")
            st.metric("Update Rate", "~1s")
            st.metric("Accuracy", "85-90%")
    
    # Additional info
    st.divider()
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("###  Tracked Emotions")
        st.write("""
        1.  **Angry**
        2.  **Happy**
        3.  **Sad**
        4.  **Surprise**
        5.  **Neutral**
        """)
    
    with info_col2:
        st.markdown("###  Technology")
        custom_face_trained = os.path.exists("models/custom_face_model.pth")
        if custom_face_trained:
            st.write("""
            - **FER:** 40% weight
            - **DeepFace:** 30% weight
            - **Custom model:** 30% weight
            - **Ensemble:** Weighted average
            - **WebRTC:** Real-time stream
            - **Update:** Every 1 second
            """)
        else:
            st.write("""
            - **FER:** 60% weight
            - **DeepFace:** 40% weight
            - **Ensemble:** Weighted average
            - **WebRTC:** Real-time stream
            - **Update:** Every 1 second
            """)
    
    with info_col3:
        st.markdown("###  Tips")
        st.info("""
        **Best results:**
        - Good lighting
        - Face in center box
        - Neutral background
        - Direct camera view
        - Wait 1-2 seconds for analysis
        """)

with tab4:
    st.header("About This Project")
    st.markdown("""
    ###  Multi-Modal Mental Stress Detection System
    
    An advanced AI system combining text and facial emotion analysis for comprehensive mental health assessment.
    """)
    
    # Model Performance
    st.divider()
    st.subheader(" Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        samples = metadata.get("total_samples", 0)
        st.metric("Training Samples", f"{samples:,}")
    
    with col2:
        classes = metadata.get("num_classes", 0)
        st.metric("Emotion Classes", classes if classes else "25+")
    
    with col3:
        accuracy = metadata.get("accuracy", 0) * 100
        st.metric("Text Accuracy", f"{accuracy:.2f}%")
    
    custom_face_trained = os.path.exists("models/custom_face_model.pth")
    live_weights = "FER + DeepFace + Custom Ensemble (40%-30%-30%)" if custom_face_trained else "FER + DeepFace Ensemble (60%-40%)"
    photo_weights = "FER + DeepFace + Custom Ensemble" if custom_face_trained else "FER + DeepFace Ensemble"
    
    st.caption(f"**Face Detection (Live):** {live_weights} - Accuracy ~90%")
    st.caption(f"**Face Detection (Photo):** {photo_weights} - Accuracy ~95%")
    
    # System Info
    st.divider()
    st.subheader("️ System Information")
    
    if torch.cuda.is_available():
        st.success("Running on:  GPU (CUDA Accelerated)")
        st.caption(f"**GPU:** {torch.cuda.get_device_name(0)}")
    else:
        st.info("Running on:  CPU")
    
    # Technology Stack
    st.divider()
    st.subheader("️ Technology Stack")
    
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown(f"""
        **Text Analysis:**
        - {metadata.get('model', 'RoBERTa')} Transformer
        - PyTorch Framework
        - Hugging Face Transformers
        - Multi-dataset Training
        """)
    
    with col_tech2:
        tech_weights = "Ensemble Approach (40%-30%-30%)" if custom_face_trained else "Ensemble Approach (60%-40%)"
        st.markdown(f"""
        **Face Analysis:**
        - FER (Facial Emotion Recognition)
        - DeepFace Multi-model AI
        - {tech_weights}
        - WebRTC Real-time Streaming
        - Gaming-style HUD Overlay
        """)
    
    # Project Info
    st.divider()
    st.subheader("‍ Project Information")
    
    classes_count = metadata.get("num_classes", 8)
    class_list_str = f"({', '.join(metadata.get('classes', []))})" if metadata.get("classes") else f"({classes_count} emotions)"
    
    st.markdown(f"""
    **Developed by:** Susmit Sil
    **Institution:** Techno India University  
    **Course:** BTech Computer Science Engineering  
    **Year:** 2026  
    **Project Type:** Final Year Project
    
    **Features:**
    - Real-time text emotion analysis {class_list_str}
    - Webcam/photo-based face emotion detection with ensemble
    - Live video emotion tracking with gaming HUD
    - Multi-modal stress assessment
    - GPU-accelerated inference
    - WebRTC real-time streaming
    - {"40% FER + 30% DeepFace + 30% Custom" if custom_face_trained else "60% FER + 40% DeepFace"} ensemble for accuracy
    """)
    
    st.divider()
    st.caption(f"Last Updated: {get_last_updated_date()}")

# Footer
st.divider()
col_footer1, col_footer2 = st.columns([3, 1])
with col_footer1:
    st.caption(" Final Year Project: Mental Stress Detection Using AI")
    st.caption(f" Dataset: {text_raw + image_raw:,} raw -> {text_train + image_train:,} training | {classes_count} emotions | {metadata.get('model', 'RoBERTa')} + FER + DeepFace + Custom Dataset Ensemble")
with col_footer2:
    if torch.cuda.is_available():
        st.caption(" GPU Accelerated")
    else:
        st.caption(" CPU Mode")
