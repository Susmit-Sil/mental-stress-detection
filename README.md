# Multi-Modal Mental Stress Detection System

This repository contains the source code, training pipelines, and evaluation subsystems for a Multi-Modal Mental Stress Detection System. The application leverages Deep Learning and Natural Language Processing to evaluate mental stress levels using textual, image-based, and real-time video inputs.

Developed as a Final Year Project for the BTech Computer Science Engineering program (Year 2026) at Techno India University.

## Research and Development Team
* Susmit Sil
* Saimoon Parvin
* Chitrita Neogy
* Aryan Chettri

Institution: Techno India University, Kolkata

***

## System Overview

The system is split into four primary functional components:
1. Text Emotion and Suicidal Intention Pipeline: Evaluates written text to detect primary emotions and potential mental health crises.
2. Face Emotion Recognition Photo Pipeline: Analyzes static images using a deep learning model ensemble to detect user emotion.
3. Real-Time WebRTC Live Video Pipeline: Tracks facial emotions in real time using web cameras, drawing an overlay Heads-Up Display with bounding boxes and active class confidence bars.
4. Model Evaluation Subsystem: A modular test suite located under the Model-Evaluation folder that compares the active architecture against seven other transformer architectures.

***

## Core Architecture Pipelines

### 1. Text Analysis Pipeline
* Model Architecture: RoBERTa-Base (Sequence Classification format via Hugging Face Transformers).
* Metric Performance: 81.85 percent Accuracy, 81.58 percent Precision, 81.85 percent Recall, and 81.64 percent F1-Score.
* Dataset Scale: Trained on 80,000 balanced text samples.
* Target Classes: Anger, Fear, Joy, Love, Neutral, Sadness, Suicidal, Surprise.
* Dataset Integration: Aggregated from multiple raw source datasets inside data/raw (including Suicide Detection datasets, Bangla Suicidal Intention text, Reddit SW logs, student anxiety datasets, and Dell tweets). Raw records are preprocessed, cleaned, and balanced to prevent minority class bias.
* Inference Engine: Automatically tokenizes text using the RoBERTa tokenizer (max length 128/256), moving tensors to CUDA for rapid GPU-accelerated classification.
* Crisis Detection: Integrates a secondary keyword-based safety layer that intercepts emergency mental health states (e.g. suicidal ideation) and displays contact information for regional and international crisis helplines.

### 2. Face Analysis Pipeline (Photo Mode)
* Architecture: Weighted Ensemble combining FER (Facial Emotion Recognition), DeepFace, and a Custom MobileNetV2 classifier.
* Custom Model: Built using a MobileNetV2 backbone (pretrained on ImageNet), adapted for single-channel grayscale inputs, and fine-tuned on custom facial image sets.
* Weighted Logic: If a custom face model is detected:
  * FER weight: 40 percent
  * DeepFace weight: 30 percent
  * Custom model weight: 30 percent
  Otherwise, the system defaults to a baseline combination of 60 percent FER and 40 percent DeepFace.
* Target Classes: angry, happy, sad, surprise, neutral.

### 3. Live Video Pipeline (WebRTC HUD)
* Architecture: Integrates streamlit-webrtc to execute in-browser, real-time video processing.
* Frame Processor: Operates a custom VideoProcessorBase class using OpenCV. Video frames are captured in BGR format, flipped for a natural mirror effect, and routed to the face emotion ensemble.
* Overlay HUD: Renders a retro gaming HUD containing:
  * A face-tracking bounding box with a status label displaying the top detected emotion.
  * Real-time horizontal confidence graphs showing active emotion intensities.
  * System acceleration status (showing GPU or CPU execution state).

### 4. Model Evaluation Subsystem
An isolated playground located under the Model-Evaluation folder. It allows running comparative evaluations on different transformer models on multiple datasets.
* Model Architectures Compared:
  * RoBERTa
  * BERT+CNN
  * BERT
  * DistilBERT
  * MiniLM
  * ELECTRA
  * T5
  * FinBERT
* Output Deliverables:
  * CSV/Excel summary tables (saved under Model-Evaluation/results/)
  * Visualization charts including accuracy heatmaps, multi-metrics radar charts, and comparative bar graphs (saved under Model-Evaluation/plots/)

***

## Hardware Acceleration and CUDA Setup

To ensure stable, real-time performance, training is restricted to compatible NVIDIA graphics hardware (e.g. RTX 4060) utilizing PyTorch with CUDA support. 
* Training Enforcement: System scripts check for CUDA availability before execution. If PyTorch cannot access a GPU, execution is stopped with an informative warning. This prevents system hangs from CPU training.
* Inference Acceleration: In chatbot-mega.py, cudnn benchmark mode is enabled, and PyTorch inference runs inside a torch.inference_mode context block to minimize VRAM usage.

***

## Directory Structure

* data/: Contains balanced dataset files and training sources.
  * raw/: Folder containing more than 190 raw source files, including Reddit SW features, COVID support documents, student academic pressure sheets, and terrorism datasets.
* models/: Folder containing saved models and encoder files.
  * emotion_model_auto/: Target folder for the trained RoBERTa text model.
  * label_encoder_auto.pkl/: Saved sklearn label encoder.
  * text_model_metrics.json: Stores model metrics (accuracy, classes, sample count).
  * all_model_comparison_metrics.json: Compares realistic performance metrics across different architectures.
* scripts/: Contains modular helper scripts for pipeline operations.
  * check_gpu.py: Verifies PyTorch can access CUDA and prints device hardware specifications.
  * prepare_auto_dataset.py: Aggregates raw text datasets into combined and balanced formats.
  * train_auto_model.py: Executes GPU-enforced transformer training.
  * step1_crop_friends.py: Face cropping helper script.
  * step2_train_custom_model.py: Trains custom facial emotion classification network.
  * evaluate_face_model.py: Evaluates custom facial detection accuracy.
* Model-Evaluation/: Independent evaluation environment.
  * config.py: Configuration parameters (batch sizes, epochs, learning rate, target models).
  * dataset_loader.py: Data loader and splitter for evaluation tasks.
  * plotting_utils.py: Generates radar charts, heatmaps, and metric comparison charts.
  * train_and_evaluate.py: Training runner for evaluation.
* chatbot_mega.py: Main Streamlit web application orchestrator.
* run.bat: Automated batch file script to verify dependencies and start the app.
* requirements.txt: Global application requirements.

***

## Setup and Usage

### Prerequisites
* Python 3.10 or 3.11
* CUDA Toolkit (v12.1 or v12.4 matching PyTorch)
* Compatible NVIDIA Graphics Card

### Option 1: Automated Launch (Recommended)
Double-click run.bat in the project directory. The batch file will:
1. Verify if a local virtual environment (venv) exists. If not, it creates one.
2. Upgrade pip and install all requirements from requirements.txt.
3. Validate GPU/CUDA configurations.
4. Auto-generate the balanced dataset if it does not already exist.
5. Launch the Streamlit application.

### Option 2: Manual Launch
Run the following commands in order:

Create a virtual environment:
python -m venv venv

Activate the virtual environment:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Verify GPU connection:
python scripts/check_gpu.py

Start the web application:
streamlit run chatbot_mega.py

***

## Retraining the Model

If you wish to retrain the text classification model, follow these steps:
1. Delete the existing balanced dataset (data/auto_balanced_dataset.csv).
2. Delete the active model folder (models/emotion_model_auto).
3. Run the prepare script:
   python scripts/prepare_auto_dataset.py
4. Run the training script:
   python scripts/train_auto_model.py
5. The scripts will balance the dataset, run CUDA-accelerated training using RoBERTa, and write updated parameters to text_model_metrics.json.
