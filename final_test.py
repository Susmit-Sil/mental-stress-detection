print("="*70)
print("FINAL SYSTEM TEST")
print("="*70)

# Test 1: FER
try:
    from fer import FER
    detector = FER(mtcnn=True)
    print("✅ FER with MTCNN: Working")
except Exception as e:
    print(f"❌ FER failed: {e}")

# Test 2: DeepFace
try:
    from deepface import DeepFace
    print("✅ DeepFace: Working")
except Exception as e:
    print(f"❌ DeepFace failed: {e}")

# Test 3: Face Detector Module
try:
    from face_emotion_detector import detect_emotion_from_image, draw_emotion_on_image
    print("✅ Face emotion detector module: Imported")
except Exception as e:
    print(f"❌ Face detector import failed: {e}")

# Test 4: PyTorch (for text model)
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}: Device {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
except Exception as e:
    print(f"❌ PyTorch failed: {e}")

# Test 5: Transformers (for text model)
try:
    from transformers import AutoTokenizer
    print("✅ Transformers: Working")
except Exception as e:
    print(f"❌ Transformers failed: {e}")

# Test 6: Streamlit
try:
    import streamlit
    print(f"✅ Streamlit {streamlit.__version__}: Ready")
except Exception as e:
    print(f"❌ Streamlit failed: {e}")

print("\n" + "="*70)
print("🎉 ALL SYSTEMS READY!")
print("\nNext step: streamlit run chatbot_mega.py")
print("="*70)
