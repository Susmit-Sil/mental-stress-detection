# AGENT SETUP INSTRUCTIONS

This document contains automated instructions for future AI agents to rebuild the development environment and restore the workspace.

## ⚙️ Environment Overview
- **Project Type**: Machine Learning / Mental Stress Detection
- **OS**: Windows (PowerShell/CMD compatible paths)
- **Language**: Python (requires Python 3.8+)
- **Dependency File**: `requirements.txt`
- **Data Hosting**: Hugging Face datasets (`mental-stress-detection-dataset`)

## 🛠️ Step-by-Step Environment Build

### 1. Initialize Virtual Environment
Future agents should create and activate the python virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
Install all required libraries locked in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 3. Restore Archived Datasets and Models
The heavy directories (`data`, `datasets`, `models`, `Model-Evaluation`, and `results_auto`) are stored as zip files on Hugging Face to optimize git operations. 

Execute the recovery script to fetch them:
```powershell
python -c "
import os, zipfile
from huggingface_hub import hf_hub_download
token = os.environ.get('HF_TOKEN', '<YOUR_HF_TOKEN>')
repo = 'susmitsil07/mental-stress-detection-dataset'
files = ['data.zip', 'datasets.zip', 'models.zip', 'Model-Evaluation.zip', 'results_auto.zip', 'FER.zip']
for f in files:
    try:
        p = hf_hub_download(repo_id=repo, filename=f, repo_type='dataset', token=token)
        with zipfile.ZipFile(p, 'r') as zf:
            zf.extractall('.')
        print(f'Recovered: {f}')
    except Exception as e:
        print(f'Failed {f}: {e}')
"
```

## 🔍 Validation Checklist
After restoration, future agents must check that:
- [ ] `data/` exists and contains dataset files.
- [ ] `datasets/` exists and contains dataset files.
- [ ] `models/` contains trained weights/checkpoints.
- [ ] `Model-Evaluation/` is present.
- [ ] `results_auto/` is present.
- [ ] Running `python -c "import huggingface_hub; print('Ready')"` succeeds.
