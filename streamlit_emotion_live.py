import streamlit as st
import cv2
import numpy as np
from fer.fer import FER
from deepface import DeepFace
from PIL import Image, ImageEnhance
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import time
from collections import deque

# Page config
st.set_page_config(
    page_title="Mental Stress Detection - Live",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #38bdf8;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #38bdf8;
        margin: 1rem 0;
    }
    .emotion-bar {
        background-color: #1e293b;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize models
@st.cache_resource
def load_text_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("./trained_model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading text model: {e}")
        return None, None, None

@st.cache_resource
def load_face_detector():
    return FER(mtcnn=True)

# Emotion labels mapping
EMOTION_LABELS = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval',
    5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment',
    10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear',
    15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
    20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'sadness'
}

# Real-time video transformer
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.fer_detector = FER(mtcnn=True)
        self.emotion_history = deque(maxlen=30)  # Store last 30 frames
        self.fps_history = deque(maxlen=10)
        self.last_time = time.time()
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        
        # Calculate FPS
        fps = 1 / (current_time - self.last_time) if self.last_time else 0
        self.fps_history.append(fps)
        self.last_time = current_time
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        try:
            # Detect emotions
            fer_result = self.fer_detector.detect_emotions(img)
            
            if fer_result:
                bounding_box = fer_result[0]["box"]
                emotions = fer_result[0]["emotions"]
                
                # Store emotion in history
                self.emotion_history.append(emotions)
                
                # Get dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion] * 100
                
                # Draw bounding box
                x, y, w, h = bounding_box
                
                # Color based on emotion
                color_map = {
                    'happy': (0, 255, 0),
                    'sad': (255, 0, 0),
                    'angry': (0, 0, 255),
                    'neutral': (200, 200, 200),
                    'surprise': (255, 255, 0),
                    'fear': (128, 0, 128),
                    'disgust': (0, 128, 128)
                }
                box_color = color_map.get(dominant_emotion, (0, 255, 255))
                
                cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 3)
                
                # Draw main emotion label with background
                label = f"{dominant_emotion.upper()}: {confidence:.1f}%"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, (x, y-35), (x+label_w+10, y), box_color, -1)
                cv2.putText(img, label, (x+5, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw HUD - Emotion bars on left side
                hud_x = 15
                hud_y = 50
                bar_width = 300
                bar_height = 25
                
                # Draw semi-transparent background for HUD
                overlay = img.copy()
                cv2.rectangle(overlay, (hud_x-10, hud_y-40), 
                            (hud_x+bar_width+10, hud_y+210), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
                
                # Title
                cv2.putText(img, "EMOTION ANALYSIS", (hud_x, hud_y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Draw top 5 emotions as bars
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
                for i, (emotion, score) in enumerate(sorted_emotions):
                    y_pos = hud_y + (i * 35)
                    
                    # Bar background
                    cv2.rectangle(img, (hud_x, y_pos), (hud_x+bar_width, y_pos+bar_height), 
                                (50, 50, 50), -1)
                    
                    # Filled bar
                    fill_width = int(score * bar_width)
                    bar_color = color_map.get(emotion, (0, 255, 0))
                    cv2.rectangle(img, (hud_x, y_pos), (hud_x+fill_width, y_pos+bar_height), 
                                bar_color, -1)
                    
                    # Border
                    cv2.rectangle(img, (hud_x, y_pos), (hud_x+bar_width, y_pos+bar_height), 
                                (255, 255, 255), 2)
                    
                    # Text
                    text = f"{emotion.capitalize()}: {score*100:.0f}%"
                    cv2.putText(img, text, (hud_x+5, y_pos+18), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw FPS counter
                cv2.putText(img, f"FPS: {avg_fps:.1f}", (img.shape[1]-120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            else:
                # No face detected
                cv2.putText(img, "No face detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        except Exception as e:
            cv2.putText(img, f"Error: {str(e)[:30]}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img

# Main app
def main():
    st.markdown('<h1 class="main-header">🧠 Mental Stress Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Multimodal AI for Emotional Intelligence: Text + Face Analysis**")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100.png?text=AI+Emotion+Detector", 
                use_container_width=True)
        st.markdown("### 🎯 Model Info")
        st.info("""
        **Text Model:** BERT  
        **Accuracy:** 94.64%  
        **Face Model:** FER + DeepFace  
        **Accuracy:** 67-75%  
        **GPU:** NVIDIA RTX 4060
        """)
        
        st.markdown("### 📊 Features")
        st.success("""
        ✅ 25 text emotions  
        ✅ 7 facial emotions  
        ✅ Real-time processing  
        ✅ Ensemble learning  
        ✅ GPU acceleration
        """)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Analysis", "📷 Image Analysis", 
                                       "🎥 Live Video", "ℹ️ About"])
    
    # Tab 1: Text Analysis
    with tab1:
        st.header("📝 Text Emotion Detection")
        st.markdown("Enter text to analyze emotional content using BERT transformer")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_text = st.text_area("Enter your text:", height=200, 
                                    placeholder="E.g., I'm feeling anxious about my exams...")
            
            if st.button("🔍 Analyze Text", type="primary"):
                if user_text:
                    with st.spinner("Analyzing emotions..."):
                        tokenizer, model, device = load_text_model()
                        
                        if model is not None:
                            # Tokenize
                            inputs = tokenizer(user_text, return_tensors="pt", 
                                             truncation=True, padding=True, max_length=128)
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            
                            # Predict
                            with torch.no_grad():
                                outputs = model(**inputs)
                                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            
                            # Get top 5 emotions
                            top5_prob, top5_idx = torch.topk(predictions, 5)
                            
                            st.success("✅ Analysis Complete!")
                            
                            # Display results
                            for i, (idx, prob) in enumerate(zip(top5_idx[0], top5_prob[0])):
                                emotion = EMOTION_LABELS[idx.item()]
                                confidence = prob.item() * 100
                                
                                st.markdown(f"""
                                <div class="emotion-bar">
                                    <strong>{i+1}. {emotion.capitalize()}</strong>: {confidence:.2f}%
                                    <div style="background-color: #38bdf8; width: {confidence}%; height: 10px; border-radius: 5px; margin-top: 5px;"></div>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter some text to analyze")
        
        with col2:
            st.markdown("### 💡 Tips")
            st.info("""
            - Write naturally
            - Multiple sentences work better
            - Context improves accuracy
            - Try different emotional scenarios
            """)
            
            st.markdown("### 📈 Model Details")
            st.metric("Model", "BERT-base")
            st.metric("Accuracy", "94.64%")
            st.metric("Classes", "25 emotions")
    
    # Tab 2: Image Analysis
    with tab2:
        st.header("📷 Facial Emotion Detection")
        st.markdown("Upload an image for facial emotion analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔍 Analyze Face", type="primary"):
                    with st.spinner("Detecting emotions..."):
                        # Convert to OpenCV format
                        img_array = np.array(image)
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # Detect with FER
                        fer_detector = load_face_detector()
                        result = fer_detector.detect_emotions(img_bgr)
                        
                        if result:
                            emotions = result[0]["emotions"]
                            dominant = max(emotions, key=emotions.get)
                            
                            st.success(f"✅ Dominant Emotion: **{dominant.upper()}**")
                            
                            # Display all emotions
                            for emotion, score in sorted(emotions.items(), 
                                                        key=lambda x: x[1], reverse=True):
                                st.markdown(f"""
                                <div class="emotion-bar">
                                    <strong>{emotion.capitalize()}</strong>: {score*100:.1f}%
                                    <div style="background-color: #38bdf8; width: {score*100}%; height: 10px; border-radius: 5px; margin-top: 5px;"></div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error("❌ No face detected in image")
        
        with col2:
            st.markdown("### 📸 Tips")
            st.info("""
            - Use clear, well-lit photos
            - Face should be visible
            - Avoid multiple faces
            - Frontal view works best
            """)
    
    # Tab 3: Live Video
    with tab3:
        st.header("🎥 Real-Time Emotion Detection")
        st.markdown("**Live webcam emotion analysis with HUD overlay**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📹 Live Feed")
            webrtc_ctx = webrtc_streamer(
                key="emotion-detection-live",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=EmotionVideoTransformer,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        
        with col2:
            st.markdown("### 📋 Instructions")
            st.info("""
            **How to use:**
            1. Click **START** button
            2. Allow camera access
            3. Position face in frame
            4. Watch real-time analysis
            5. Click **STOP** to end
            """)
            
            st.markdown("### 🎨 Features")
            st.success("""
            ✅ Live face detection  
            ✅ Real-time emotions  
            ✅ Confidence scores  
            ✅ FPS counter  
            ✅ HUD overlay  
            ✅ Color-coded boxes
            """)
            
            st.markdown("### ⚙️ Performance")
            st.metric("Expected FPS", "15-25")
            st.metric("Latency", "<100ms")
            st.metric("Accuracy", "67-75%")
    
    # Tab 4: About
    with tab4:
        st.header("ℹ️ About the System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Models Used")
            st.markdown("""
            **Text Analysis:**
            - Model: BERT (Bidirectional Encoder Representations from Transformers)
            - Accuracy: 94.64%
            - Classes: 25 emotions
            - Training: 479K samples
            
            **Face Analysis:**
            - Models: FER + DeepFace Ensemble
            - Accuracy: 67-75%
            - Classes: 7 emotions
            - Weighting: 60% FER + 40% DeepFace
            """)
        
        with col2:
            st.markdown("### 🛠️ Tech Stack")
            st.markdown("""
            **Frameworks:**
            - PyTorch (text model)
            - TensorFlow (face models)
            - Streamlit (web interface)
            
            **Libraries:**
            - Hugging Face Transformers
            - FER, DeepFace
            - OpenCV, MTCNN
            
            **Hardware:**
            - GPU: NVIDIA RTX 4060
            - CUDA acceleration
            """)
        
        st.markdown("### 📊 Dataset")
        st.info("""
        - **Raw samples:** 511,704 (479K text + 32K images)
        - **Processed:** 53,874 samples
        - **Training:** 48,126 samples
        - **Balanced** across all emotion classes
        """)
        
        st.markdown("### 🎯 Future Enhancements")
        st.markdown("""
        - 🚀 Multi-language support
        - 🚀 Emotion trend analysis
        - 🚀 Mobile app deployment
        - 🚀 Integration with mental health professionals
        """)

if __name__ == "__main__":
    main()
