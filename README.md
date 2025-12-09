# 💬 Twitter Sentiment Analysis — OIBSIP Data Analytics Task

This project demonstrates sentiment classification on Twitter data using both classical machine learning models and modern transformer-based deep learning (DistilBERT). It was completed as part of the **Oasis Infobyte Data Analytics Internship**.

---

## 📌 Objective

To build and evaluate models that classify tweets into one of three sentiment classes:
- **Negative (-1)**
- **Neutral (0)**
- **Positive (1)**

The goal is to compare traditional ML models with a fine-tuned transformer for real-world text classification tasks.

---

## 🧾 Dataset

- **Source**: Twitter sentiment dataset (pre-cleaned or raw, depending on implementation)
- **Content**:
  - Tweet text
  - Sentiment labels (`-1`, `0`, `1`)

---

## ⚙️ Models Used

### Classical Machine Learning
- Logistic Regression
- Naive Bayes (MultinomialNB)

### Deep Learning
- DistilBERT (fine-tuned using Hugging Face Transformers + TensorFlow)

---

## 🔍 Workflow Overview

1. **Data Cleaning & Preprocessing**
   - Removed unwanted characters, stopwords, and links
   - Tokenized and lowercased text

2. **Feature Extraction**
   - TF-IDF vectorization for classical models
   - Tokenization using DistilBERT tokenizer for deep learning

3. **Model Training**
   - Trained ML models on vectorized tweets
   - Fine-tuned DistilBERT for multi-class classification

4. **Evaluation**
   - Compared accuracy, precision, recall, and F1-score
   - Plotted confusion matrices

---

## 🧰 Tools & Libraries

- Python
- Scikit-learn
- Hugging Face Transformers
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib

---

## 📈 Results

- Classical models performed well on basic features
- DistilBERT achieved significantly higher accuracy and generalization
- Fine-tuned BERT showed best performance on nuanced tweets

---
