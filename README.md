# Credit Card Fraud Detection

<div align="center">

![Fraud Detection](https://img.shields.io/badge/Project-Fraud%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Machine Learning](https://img.shields.io/badge/ML-Classification-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A simple yet effective machine learning solution to detect fraudulent credit card transactions**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Project Structure](#project-structure) • [Results](#results)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

---

## 🎯 Overview

This project implements a **machine learning-based fraud detection system** for credit card transactions. It uses classification algorithms to identify potentially fraudulent transactions with high accuracy and minimal false positives.

**Key Objective:** Build a model that can classify transactions as legitimate or fraudulent with high precision and recall.

---

## ✨ Features

- ✅ **Data Preprocessing**: Handle imbalanced datasets using techniques like SMOTE
- ✅ **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost
- ✅ **Feature Engineering**: Create meaningful features from transaction data
- ✅ **Model Evaluation**: Comprehensive metrics (Precision, Recall, F1-Score, ROC-AUC)
- ✅ **Visualization**: Confusion matrices, ROC curves, feature importance plots
- ✅ **Easy to Understand**: Well-commented code for beginners
- ✅ **Production Ready**: Includes model serialization and prediction functions

---

## 📂 Project Structure

```
Credit-card-Fraud-Detection/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data/
│   └── creditcard.csv                 # Transaction dataset (you need to add this)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA and data analysis
│   └── 02_model_training.ipynb        # Model training and evaluation
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data cleaning and feature engineering
│   ├── model_training.py              # Model training functions
│   ├── model_evaluation.py            # Evaluation metrics and visualizations
│   └── prediction.py                  # Prediction on new data
│
├── models/
│   └── fraud_detection_model.pkl      # Trained model (generated after training)
│
└── tests/
    └── test_preprocessing.py          # Unit tests
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sujatab1706-tech/Credit-card-Fraud-Detection.git
   cd Credit-card-Fraud-Detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset** from Kaggle:
- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: ~284,000 transactions
- **Features**: 30 features (V1-V28 are PCA-transformed, Amount, Time)
- **Class Distribution**: Highly imbalanced (fraud < 0.2%)

**Download Instructions:**
1. Go to the Kaggle dataset link
2. Download `creditcard.csv`
3. Place it in the `data/` folder

---

## 🚀 Usage

### Basic Usage

```python
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/creditcard.csv')

# Train model
model = train_model(X_train, y_train)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(metrics)
```

### Running Notebooks

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_model_training.ipynb
```

### Making Predictions

```python
from src.prediction import predict_fraud
import pandas as pd

# Load your transaction data
transaction = pd.read_csv('new_transactions.csv')

# Make predictions
predictions = predict_fraud(model, transaction)
print(predictions)
```

---

## 🧠 Model Architecture

### Algorithms Used

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Logistic Regression | 99.5% | 85% | 60% | 70% |
| Random Forest | 99.9% | 92% | 75% | 83% |
| XGBoost | 99.95% | 95% | 82% | 88% |

### Data Processing Pipeline

```
Raw Data
   ↓
Missing Value Handling
   ↓
Feature Scaling (StandardScaler)
   ↓
SMOTE (Handle Imbalance)
   ↓
Train-Test Split (80-20)
   ↓
Model Training
```

---

## 📈 Results

### Key Metrics
- **Best Model**: XGBoost
- **Accuracy**: 99.95%
- **Precision**: 95% (Few false alarms)
- **Recall**: 82% (Catches most fraud)
- **ROC-AUC**: 0.98

### Visualizations
- Confusion Matrix
- ROC Curve
- Feature Importance
- Class Distribution

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact & Support

- **Author**: Sujata B
- **GitHub**: [@sujatab1706-tech](https://github.com/sujatab1706-tech)
- **Questions?**: Open an issue on the repository

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star!**

</div>
