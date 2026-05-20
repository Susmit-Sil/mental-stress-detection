import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os

def is_cloud_environment():
    """Detect if running on Streamlit Cloud"""
    # Streamlit Cloud sets this environment variable
    return os.getenv('STREAMLIT_SHARING', 'false') == 'true' or \
           os.path.exists('/mount/src/')  # Cloud mount path

def get_detector_version():
    """Determine which detector to use"""
    if is_cloud_environment():
        return 'cloud'
    
    # Try to import FER for local
    try:
        import fer
        return 'local'
    except ImportError:
        return 'cloud'

# Global config
DETECTOR_VERSION = get_detector_version()
USE_FER = DETECTOR_VERSION == 'local'

print(f"Environment: {'CLOUD' if is_cloud_environment() else 'LOCAL'}")
print(f"Detector: {'FER + DeepFace (95-97%)' if USE_FER else 'DeepFace only (93-95%)'}")
