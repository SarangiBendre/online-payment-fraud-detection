# 💳 Online Payment Fraud Detection System

An end-to-end Machine Learning project that detects fraudulent online payment transactions using **XGBoost** and provides predictions through a **Streamlit-based user interface**.

---

## 📌 Project Overview

Online payment systems face significant financial risks due to fraudulent transactions.  
This project builds a **fraud detection model** using supervised machine learning and deploys it as a **web application** for real-time prediction.

The model is trained on a highly **imbalanced dataset**, reflecting real-world payment data, and focuses on **fraud recall and risk scoring** rather than just accuracy.

---

## 🎯 Objectives

- Detect fraudulent online payment transactions
- Handle extreme class imbalance
- Build a complete ML pipeline (training → evaluation → inference)
- Provide a usable **web-based user interface**
- Deploy the application publicly

---

## 📊 Dataset

- Source: Online Payment Fraud Dataset (Kaggle)
- Size: ~165,000 transactions
- Fraud cases: Extremely rare (< 0.2%)

### Key Features Used
- Transaction amount
- Sender & receiver balances (before and after transaction)
- Transaction type (TRANSFER, CASH_OUT, etc.)
- Engineered balance difference features
- System-generated transaction step

> ⚠️ The raw dataset is **not included** in this repository due to size limitations.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Removed non-informative ID columns
- Handled missing values
- Performed one-hot encoding for transaction types

### 2️⃣ Feature Engineering
- Balance difference for sender and receiver
- Internal system features such as transaction step

### 3️⃣ Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** on training data

### 4️⃣ Model Training
- Algorithm: **XGBoost Classifier**
- Tuned for better fraud recall
- Evaluated using:
  - Confusion Matrix
  - Precision, Recall, F1-score
  - ROC-AUC

### 5️⃣ Threshold Optimization
- Custom probability threshold to reduce missed fraud cases
- Business-oriented evaluation approach

---

## 📈 Results

- ROC-AUC: ~0.97
- High fraud recall
- Low false-negative rate
- Risk score provided instead of only binary output

---

## 🖥️ User Interface

- Built using **Streamlit**
- Accepts transaction details as input
- Displays:
  - Fraud / Legitimate decision
  - Fraud probability (risk score)

The UI simulates how a **backend fraud detection engine** would work in real payment systems.

---

## 🚀 Deployment

- Deployed using **Render**
- Application runs as a public web service
- Model loaded from serialized `.pkl` file

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Streamlit
- GitHub
- Render

---

## 🧪 How to Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py


Open browser:

http://localhost:8501

## 🧠 Key Learnings

Handling imbalanced datasets in real-world ML problems

Importance of feature consistency between training and inference

Practical ML deployment challenges

Building usable ML systems, not just models

## 🔮 Future Improvements

Cost-sensitive learning

Real-time transaction streaming

Model monitoring and drift detection

Authentication and role-based UI

---
##👤 Author

Sarangi
B.Tech – Artificial Intelligence & Machine Learning

📌 Conclusion

This project demonstrates a complete, real-world machine learning pipeline for fraud detection, from data preprocessing to deployment with a functional user interface.


---
