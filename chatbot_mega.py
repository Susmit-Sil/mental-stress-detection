import streamlit as st
import torch

torch.backends.cudnn.benchmark = True  # ✅ speed boost for fixed input sizes

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json
import time


st.set_page_config(
    page_title="Mental Stress Detection AI", 
    page_icon="🧠",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained('./emotion_model_auto')
    model = AutoModelForSequenceClassification.from_pretrained('./emotion_model_auto')
    
    model = model.to(device)   # ✅ MOVE MODEL TO DEVICE
    
    label_encoder = pickle.load(open('label_encoder_auto.pkl', 'rb'))
    
    try:
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
    except:
        metadata = {}
    
    model.eval()  # ✅ inference mode (recommended)
    
    return model, tokenizer, label_encoder, device, metadata


with st.spinner("🔄 Loading AI model..."):
    model, tokenizer, label_encoder, device, metadata = load_model()

# Sidebar
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.metric("Training Samples", f"{metadata.get('total_samples', 'N/A'):,}")
    st.metric("Emotion Classes", metadata.get('num_classes', 'N/A'))
    st.metric("Model Accuracy", f"{metadata.get('accuracy', 0)*100:.1f}%")
    
    gpu_status = "🟢 GPU" if torch.cuda.is_available() else "🔴 CPU"
    st.success(f"**Running on:** {gpu_status}")
    
    if torch.cuda.is_available():
        st.info(f"**GPU:** {torch.cuda.get_device_name(0)}")
    
    st.divider()
    st.subheader("🛠️ Technology Stack")
    st.write("- DistilBERT Transformer")
    st.write("- PyTorch (GPU Accelerated)")
    st.write("- Multi-dataset Training")
    st.write("- 3 Combined Datasets")
    
    st.divider()
    st.caption("📚 Datasets Used:")
    st.caption("• Emotion Dataset (HF)")
    st.caption("• GoEmotions (Google)")
    st.caption("• Custom Sentiment Data")

# Main UI
st.title("🧠 Mental Stress Detection AI Chatbot")
st.write("Advanced emotion detection powered by 500K+ training samples")

# Input
st.subheader("💭 Share Your Thoughts")
user_input = st.text_area(
    "How are you feeling today?",
    height=180,
    placeholder="Express your thoughts, feelings, or experiences...\n\nExamples:\n- I'm worried about my family's safety\n- Today was absolutely amazing!\n- I feel anxious about my upcoming exams"
)

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
with col_btn1:
    analyze_btn = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)

# Analysis
if analyze_btn:
    if user_input.strip() and len(user_input) > 5:
        with st.spinner("🤖 Analyzing with AI..."):
            start_time = time.time()
            
            # Tokenize (creates CPU tensors by default)
            inputs = tokenizer(
                user_input, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            
            # CRITICAL FIX: Move ALL tensors to same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.inference_mode():   # even faster than no_grad
                outputs = model(**inputs)

                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get predicted class
                predicted_class = torch.argmax(predictions, dim=1).item()
                confidence = predictions[0][predicted_class].item() * 100
                
                # Get top 5 predictions
                top5_probs, top5_indices = torch.topk(predictions[0], min(5, len(label_encoder.classes_)))
                top5_emotions = [label_encoder.inverse_transform([idx.item()])[0] 
                               for idx in top5_indices]
                top5_confidences = [prob.item() * 100 for prob in top5_probs]
            
            processing_time = time.time() - start_time
            
            # Decode primary emotion
            emotion = label_encoder.inverse_transform([predicted_class])[0]
            
            # Display results
            st.divider()
            st.subheader("🎯 Analysis Results")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Primary Emotion", emotion.upper())
            with col_b:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col_c:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            # Top 5 emotions chart
            st.write("**Top 5 Detected Emotions:**")
            for i, (emo, conf) in enumerate(zip(top5_emotions, top5_confidences), 1):
                st.progress(conf/100, text=f"{i}. {emo}: {conf:.1f}%")
            
            # Mental health assessment
            st.divider()
            st.subheader("🏥 Mental Health Assessment")
            
            stress_emotions = [
                'anger', 'anxiety', 'annoyance', 'confusion', 'disappointment',
                'disapproval', 'disgust', 'embarrassment', 'fear', 'grief',
                'nervousness', 'remorse', 'sadness', 'negative', 'stress',
                'worry', 'frustrated', 'overwhelmed'
            ]
            
            positive_emotions = [
                'joy', 'love', 'admiration', 'amusement', 'approval', 'caring',
                'curiosity', 'desire', 'excitement', 'gratitude', 'optimism',
                'pride', 'relief', 'surprise', 'happiness', 'positive'
            ]
            
            emotion_lower = emotion.lower()
            
            if any(s in emotion_lower for s in stress_emotions):
                st.error("⚠️ **Elevated Stress Level Detected**")
                
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    st.write("**Immediate Actions:**")
                    st.write("✅ Take 10 deep breaths (4-7-8 technique)")
                    st.write("✅ Step away for 5-10 minutes")
                    st.write("✅ Drink water and stretch")
                    
                with col_rec2:
                    st.write("**Helpful Resources:**")
                    st.write("💬 Talk to a trusted friend/family")
                    st.write("🧘 Try meditation apps (Headspace, Calm)")
                    st.write("📝 Journal your thoughts")
                
                st.warning("💡 If feelings persist, consider speaking with a mental health professional")
                    
            elif any(p in emotion_lower for p in positive_emotions):
                st.success("✅ **Positive Mental State Detected**")
                st.write("🌟 Great! Your emotional well-being seems positive.")
                st.write("💪 Continue with:")
                col_pos1, col_pos2 = st.columns(2)
                with col_pos1:
                    st.write("• Maintain healthy sleep schedule")
                    st.write("• Stay physically active")
                with col_pos2:
                    st.write("• Connect with loved ones")
                    st.write("• Practice gratitude")
                
            else:
                st.info("ℹ️ **Neutral Emotional State**")
                st.write("Your emotional state appears balanced. Stay mindful and check in with yourself regularly.")
                
    elif user_input.strip():
        st.warning("⚠️ Please enter at least 5 characters for accurate analysis")
    else:
        st.warning("⚠️ Please enter some text to analyze!")


# Footer
st.divider()
st.caption("🎓 Final Year Project: Mental Stress Detection Using AI")
st.caption(f"📊 Trained on {metadata.get('total_samples', 'N/A'):,} samples | {metadata.get('num_classes', 'N/A')} emotion classes | Accuracy: {metadata.get('accuracy', 0)*100:.1f}%")
st.caption("💻 BTech Computer Science Engineering | Powered by DistilBERT & PyTorch")
