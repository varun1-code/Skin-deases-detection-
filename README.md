# Skin-deases-detection-
This project uses deep learning (CNN) to classify skin lesions from the **HAM10000 dataset** into seven different categories.   It is designed to assist in early detection of skin abnormalities and can be extended for real-world clinical applications.
# Skin Disease Detection MVP (CareBot)

A deep learning-based system that detects 7 types of skin diseases using ensemble learning (EfficientNet-B0, B3, and ResNet50) and a meta-learner.  
The app provides predictions and basic care guidance via a Streamlit frontend.

## Project Workflow
1. Data Preparation (HAM10000)
2. Model Training (EfficientNet, ResNet)
3. Ensemble & Stacking
4. TorchScript Export
5. API + Streamlit Deployment

## Tech Stack
- Python, PyTorch, FastAPI, Streamlit
- Models: EfficientNet-B0, EfficientNet-B3, ResNet50
- Dataset: HAM10000

## Results
- Ensemble Accuracy: **87.3%**
- Stacked Meta-Learner Accuracy: **≈ 87.3%**
- Test set: 1503 images, 7 classes

## Run Instructions
```bash
# backend API
uvicorn app:app --host 0.0.0.0 --port 8000

# frontend
streamlit run app_streamlit_custom.py
