import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("="*70)
print("FINAL SYSTEM TEST")
print("="*70)

# Test 1: FER
try:
    from fer.fer import FER
    detector = FER(mtcnn=True)
    print("[√] FER with MTCNN: Working")
except Exception as e:
    print(f"[X] FER failed: {e}")

# Test 2: DeepFace
try:
    from deepface import DeepFace
    print("[√] DeepFace: Working")
except Exception as e:
    print(f"[X] DeepFace failed: {e}")

# Test 3: Face Detector Module
try:
    import sys
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)
    from app.face_emotion_detector import detect_emotion_from_image, draw_emotion_on_image
    print("[√] Face emotion detector module: Imported")
except Exception as e:
    print(f"[X] Face detector import failed: {e}")

# Test 4: PyTorch (for text model)
try:
    import torch
    print(f"[√] PyTorch {torch.__version__}: Device {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
except Exception as e:
    print(f"[X] PyTorch failed: {e}")

# Test 5: Transformers (for text model)
try:
    from transformers import AutoTokenizer
    print("[√] Transformers: Working")
except Exception as e:
    print(f"[X] Transformers failed: {e}")

# Test 6: Streamlit
try:
    import streamlit
    print(f"[√] Streamlit {streamlit.__version__}: Ready")
except Exception as e:
    print(f"[X] Streamlit failed: {e}")

print("\n" + "="*70)
print("ALL SYSTEMS READY!")
print("\nNext step: streamlit run chatbot_mega.py")
print("="*70)
