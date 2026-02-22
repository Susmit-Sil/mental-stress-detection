import torch
import sys

def check_env_gpu():
    print(f"--- Environment: {sys.prefix} ---")
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Check if PyTorch is installed first
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        
        # Check for RTX 4060 Detection
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_id)
            print(f"[✓] GPU DETECTED: {gpu_name}")
            print(f"    CUDA Version: {torch.version.cuda}")
            print(f"    VRAM Available: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
        else:
            print("[!] GPU NOT DETECTED: Falling back to CPU.")
            print("    Reason: You likely installed the CPU version of Torch.")
            
    except ImportError:
        print("[!] PyTorch is not installed in this venv.")

if __name__ == "__main__":
    check_env_gpu()