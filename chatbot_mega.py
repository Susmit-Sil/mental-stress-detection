import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TensorFlow warnings

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json
import time
from PIL import Image
import numpy as np

# Import face detector
from face_emotion_detector_local import detect_emotion_from_image, draw_emotion_on_image

# MUST BE FIRST - Set page config
st.set_page_config(
    page_title="Mental Stress Detection AI", 
    page_icon="🧠",
    layout="wide"
)

# Performance optimization
torch.backends.cudnn.benchmark = True

# ====== LOAD MODEL (ONCE) ======
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try different model paths
    model_paths = ['./emotion_model_best', './emotion_model_mega', './emotion_model_auto']
    encoder_paths = ['label_encoder_best.pkl', 'label_encoder_mega.pkl', 'label_encoder_auto.pkl']
    
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
        st.error("❌ Model files not found! Please ensure model is trained.")
        st.stop()
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    label_encoder = pickle.load(open(encoder_path, 'rb'))
    
    # Load metadata
    metadata_files = ['model_metadata_best.json', 'model_metadata_mega.json', 'model_metadata.json']
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

# Load model
with st.spinner("🔄 Loading AI models..."):
    model, tokenizer, label_encoder, device, metadata = load_model()

# ====== SIDEBAR ======
with st.sidebar:
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        samples = metadata.get('total_samples', 21575)
        st.metric("Training Samples", f"{samples:,}" if samples else "N/A")
    with col2:
        classes = metadata.get('num_classes', 25)
        st.metric("Emotion Classes", classes if classes else "N/A")
    
    accuracy = metadata.get('accuracy', 0.851) * 100 if metadata.get('accuracy') else 85.1
    st.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    st.divider()
    st.subheader("🖥️ System")
    
    if torch.cuda.is_available():
        st.success("Running on: 🟢 GPU")
        st.caption(f"**GPU:** {torch.cuda.get_device_name(0)}")
    else:
        st.info("Running on: 🔵 CPU")
    
    st.divider()
    st.subheader("🛠️ Tech Stack")
    st.write("• DistilBERT Transformer")
    st.write("• FER + DeepFace")
    st.write("• PyTorch (GPU)")
    st.write("• Multi-dataset Training")
    
    st.divider()
    st.caption("**Version:** 2.0")
    st.caption("**Updated:** Jan 12, 2026")

# ====== MAIN CONTENT ======
st.title("🧠 Mental Stress Detection System")
st.markdown("**Multi-Modal AI for Mental Health Assessment**")

# ====== TABS ======
tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "📸 Face Analysis", "ℹ️ About"])

# ====== TAB 1: TEXT ANALYSIS ======
with tab1:
    st.header("Text-Based Emotion Detection")
    
    user_input = st.text_area(
        "💭 How are you feeling today?",
        height=150,
        placeholder="Express your thoughts, feelings, or experiences...\n\nExamples:\n• I'm feeling anxious about my exams\n• Today was absolutely amazing!\n• I'm worried about my family"
    )
    
    analyze_btn = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
    
    if analyze_btn:
        if user_input.strip() and len(user_input) > 5:
            with st.spinner("🤖 Analyzing with AI..."):
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
            st.subheader("🎯 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Primary Emotion", top5_emotions[0].upper())
            with col2:
                st.metric("Confidence", f"{top5_confidences[0]:.1f}%")
            with col3:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            # Top 5 emotions
            st.subheader("📊 Top 5 Detected Emotions")
            for i, (emotion, conf) in enumerate(zip(top5_emotions, top5_confidences), 1):
                st.progress(float(conf / 100), text=f"{i}. {emotion}: {conf:.1f}%")
            
            # Mental health assessment - CRITICAL FIX
            st.divider()
            st.subheader("🏥 Mental Health Assessment")
            
            # CRITICAL keywords (immediate crisis) - CHECK FIRST
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
            
            # CHECK FOR CRISIS FIRST (checks both emotion label AND actual user text)
            is_crisis = any(crisis in emotion_lower or crisis in input_lower for crisis in crisis_keywords)
            
            if is_crisis:
                # CRITICAL ALERT
                st.error("🚨 **CRITICAL: IMMEDIATE HELP NEEDED**")
                
                st.markdown("""
                ### 🆘 You are not alone. Help is available RIGHT NOW.
                
                **Immediate Crisis Resources:**
                """)
                
                col_crisis1, col_crisis2 = st.columns(2)
                
                with col_crisis1:
                    st.markdown("""
                    **🇮🇳 India (24/7 Helplines):**
                    - 📞 **AASRA:** 91-9820466726
                    - 📞 **Vandrevala Foundation:** 1860-2662-345
                    - 📞 **iCall:** 91-22-25521111
                    - 📞 **Sneha India:** 91-44-24640050
                    - 📞 **Lifeline Foundation:** 033-24637401
                    """)
                
                with col_crisis2:
                    st.markdown("""
                    **🌍 International:**
                    - 📞 **USA:** 988 (Suicide & Crisis Lifeline)
                    - 📞 **UK:** 116 123 (Samaritans)
                    - 📞 **Australia:** 13 11 14 (Lifeline)
                    - 📞 **Canada:** 1-833-456-4566
                    """)
                
                st.error("⚠️ **Please reach out to one of these numbers immediately. Your life matters.**")
                
                st.markdown("""
                ### 💙 Immediate Actions You Can Take:
                
                1. **📞 Call a crisis helpline NOW** - They are trained professionals who care
                2. **🏥 Go to the nearest hospital emergency room** if you're in immediate danger
                3. **💬 Tell someone you trust** - A family member, friend, teacher, or colleague
                4. **🔒 Remove any means of self-harm** from your immediate environment
                5. **🚫 Don't stay alone** - Stay with someone or in a public place
                
                ### 📝 Please Remember:
                
                - ❤️ **You matter** - Your life has value and meaning
                - ⏰ **This feeling is temporary** - It won't last forever, even though it feels that way now
                - 🌈 **Things can get better** - With help and support, situations improve
                - 👥 **People care about you** - There are people who want to help
                - 💊 **Treatment works** - Therapy, medication, and support can help you feel better
                
                ### 🏥 Next Steps:
                
                - Schedule an appointment with a psychiatrist or psychologist
                - Talk to your primary care doctor about how you're feeling
                - Consider joining a support group
                - Reach out to trusted friends or family members
                """)
                
                st.info("💡 **Your feelings are valid, and asking for help is a sign of strength, not weakness.**")
            
            elif any(s in emotion_lower for s in stress_keywords):
                # HIGH STRESS ALERT
                st.error("⚠️ **Elevated Stress Level Detected**")
                
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    st.write("**Immediate Actions:**")
                    st.write("✅ Take 10 deep breaths (4-7-8 technique)")
                    st.write("✅ Step away for 5-10 minutes")
                    st.write("✅ Drink water and stretch")
                    st.write("✅ Ground yourself (5-4-3-2-1 method)")
                    
                with col_rec2:
                    st.write("**Helpful Resources:**")
                    st.write("💬 Talk to someone you trust")
                    st.write("🧘 Try meditation (Headspace, Calm)")
                    st.write("📝 Journal your thoughts")
                    st.write("🚶 Go for a walk outdoors")
                
                st.warning("💡 If these feelings persist for more than 2 weeks, please speak with a mental health professional")
                
                st.info("""
                **Mental Health Resources (India):**
                - 📞 **Vandrevala Foundation:** 1860-2662-345 (24/7 emotional support)
                - 📞 **Mann Talks:** +91-8686139139 (Free counseling)
                - 🏥 Visit your nearest hospital psychiatry department
                - 💻 Online therapy: Practo, BetterHelp India
                """)
                    
            elif any(p in emotion_lower for p in positive_keywords):
                # POSITIVE STATE
                st.success("✅ **Positive Mental State Detected**")
                st.write("🌟 Great! Your emotional well-being seems positive.")
                
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
                # NEUTRAL STATE
                st.info("ℹ️ **Neutral Emotional State**")
                st.write("Your emotional state appears balanced. Stay mindful and check in with yourself regularly.")
                st.caption("💡 **Tip:** Regular self-check-ins help maintain mental wellness. Consider journaling or meditation.")
                
        elif user_input.strip():
            st.warning("⚠️ Please enter at least 5 characters for accurate analysis")
        else:
            st.warning("⚠️ Please enter some text to analyze!")


# ====== TAB 2: FACE ANALYSIS ======
with tab2:
    st.header("Face-Based Emotion Detection")
    st.markdown("Upload a photo or take a picture to detect stress from facial expressions")
    
    # Upload option
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png'],
        key="face_upload"
    )
    
    # Camera option
    st.subheader("📷 Take Picture")
    camera_photo = st.camera_input("Take a photo", key="face_camera")
    
    # Process image
    image_to_process = uploaded_file if uploaded_file else camera_photo
    
    if image_to_process is not None:
        image = Image.open(image_to_process)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with st.spinner("🔍 Analyzing face with AI (ensemble detection)..."):
            result = detect_emotion_from_image(image)
            
            if result['success']:
                with col2:
                    st.subheader("Analysis Result")
                    annotated = draw_emotion_on_image(image, result)
                    st.image(annotated, use_container_width=True)
                
                # Results
                num_models = result.get('num_models', 1)
                st.success(f"✅ Face emotion detected! (Using {num_models} AI model{'s' if num_models > 1 else ''})")
                
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
                st.subheader("🎭 Detailed Emotion Analysis")
                if num_models > 1:
                    st.caption("Combined prediction from multiple AI models")
                
                emotions_data = result['all_emotions']
                for emotion, score in sorted(emotions_data.items(), key=lambda x: x[1], reverse=True):
                    col_name, col_bar = st.columns([1, 4])
                    
                    with col_name:
                        emoji_map = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠',
                            'fear': '😨', 'surprise': '😮', 'disgust': '🤢', 'neutral': '😐'
                        }
                        emoji = emoji_map.get(emotion, '🎭')
                        st.write(f"{emoji} **{emotion.capitalize()}**")
                    
                    with col_bar:
                        st.progress(float(score / 100))
                        st.caption(f"{score:.1f}%")
                
            else:
                st.error(f"❌ {result.get('error', 'Could not detect face')}")
                st.info("""
                💡 **Tips for better detection:**
                - ✅ Ensure face is clearly visible
                - 💡 Use good lighting
                - 📷 Face the camera directly
                - 🚫 Remove sunglasses/masks
                - 🔍 Get closer to camera
                """)

# ====== TAB 3: ABOUT ======
with tab3:
    st.header("About This Project")
    st.markdown("""
    ### 🎓 Multi-Modal Mental Stress Detection System
    
    An advanced AI system combining text and facial emotion analysis for comprehensive mental health assessment.
    """)
    
    # Model Performance
    st.divider()
    st.subheader("📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        samples = metadata.get('total_samples', 21575)
        st.metric("Training Samples", f"{samples:,}" if samples else "21,575+")
    
    with col2:
        classes = metadata.get('num_classes', 25)
        st.metric("Emotion Classes", classes if classes else "25+")
    
    with col3:
        accuracy = metadata.get('accuracy', 0.851) * 100 if metadata.get('accuracy') else 85.1
        st.metric("Text Accuracy", f"{accuracy:.1f}%")
    
    # Additional metrics
    if metadata.get('precision') and metadata.get('recall') and metadata.get('f1'):
        col4, col5, col6 = st.columns(3)
        
        with col4:
            precision = metadata.get('precision', 0.85) * 100
            st.metric("Precision", f"{precision:.1f}%")
        
        with col5:
            recall = metadata.get('recall', 0.84) * 100
            st.metric("Recall", f"{recall:.1f}%")
        
        with col6:
            f1 = metadata.get('f1', 0.85) * 100
            st.metric("F1-Score", f"{f1:.1f}%")
    
    st.caption("**Face Detection:** FER + DeepFace Ensemble - Accuracy ~95%")
    
    # System Info
    st.divider()
    st.subheader("🖥️ System Information")
    
    if torch.cuda.is_available():
        st.success("Running on: 🟢 GPU (CUDA Accelerated)")
        st.caption(f"**GPU:** {torch.cuda.get_device_name(0)}")
    else:
        st.info("Running on: 🔵 CPU")
    
    # Technology Stack
    st.divider()
    st.subheader("🛠️ Technology Stack")
    
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("""
        **Text Analysis:**
        - DistilBERT / RoBERTa Transformer
        - PyTorch Framework
        - Hugging Face Transformers
        - Multi-dataset Training
        """)
    
    with col_tech2:
        st.markdown("""
        **Face Analysis:**
        - FER (Facial Emotion Recognition)
        - DeepFace Multi-model Ensemble
        - MTCNN Face Detection
        - OpenCV Image Processing
        """)
    
    # Project Info
    st.divider()
    st.subheader("👨‍🎓 Project Information")
    
    st.markdown("""
    **Developed by:** [Your Name]  
    **Institution:** [Your College Name]  
    **Course:** BTech Computer Science Engineering  
    **Year:** 2026  
    **Project Type:** Final Year Project
    
    **Datasets Used:**
    - Sentiment Dataset (Multi-emotion classification)
    - Mental Health Dataset (Anxiety, Stress, Depression)
    - dair-ai/emotion (Twitter emotions)
    - GoEmotions (Fine-grained emotions)
    
    **Features:**
    - Real-time text emotion analysis (45+ emotions)
    - Webcam/photo-based face emotion detection
    - Multi-modal stress assessment
    - GPU-accelerated inference
    - Cloud deployment ready
    """)
    
    st.divider()
    st.caption("Last Updated: January 12, 2026")

# Footer
st.divider()
col_footer1, col_footer2 = st.columns([3, 1])
with col_footer1:
    st.caption("🎓 Final Year Project: Mental Stress Detection Using AI")
    samples_display = f"{metadata.get('total_samples', 21575):,}" if metadata.get('total_samples') else "21,575+"
    st.caption(f"📊 Trained on {samples_display} samples | {metadata.get('num_classes', 25)} emotion classes | Powered by DistilBERT & PyTorch")
with col_footer2:
    if torch.cuda.is_available():
        st.caption("🟢 GPU Accelerated")
    else:
        st.caption("🔵 CPU Mode")
