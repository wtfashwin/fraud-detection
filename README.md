#  Project: AI-Powered Fraud Detection & Prevention System
 

This project implements a machine learning pipeline to detect fraudulent credit card transactions. It uses logistic regression on a highly imbalanced dataset, applying SMOTE to balance classes.

## Features
- Exploratory Data Analysis (EDA)
- Data preprocessing with feature scaling and oversampling (SMOTE)
- Logistic Regression model training and evaluation
- Single transaction fraud prediction script
- Visualization of class distribution and evaluation metrics

## Project Structure
- `data/` - Raw and processed datasets  
- `models/` - Trained model and scaler saved here  
- `plots/` - Visualizations like ROC curve and distribution plots  
- `preprocess.py` - Data preprocessing script  
- `train_model.py` - Model training script  
- `evaluate_model.py` - Model evaluation and plotting  
- `predict_single.py` - Single transaction prediction tool

## Requirements
- Python 3.8+  
- Packages listed in `requirements.txt`

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt


Usage
Run preprocessing:
python preprocess.py

Train model:
python train_model.py

Evaluate model:
python evaluate_model.py

Predict single transaction:
python predict_single.py

Ashwin Upadhyay


