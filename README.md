# 📰 AI-Powered Fake News Detection System

This repository contains an end-to-end Machine Learning pipeline to detect fake news using Natural Language Processing (NLP) and Logistic Regression. 

## 🎯 Objectives Achieved
1. **Machine Learning Model**: Built a Logistic Regression classifier for text analysis.
2. **Data Preprocessing**: Implemented a robust NLP pipeline (cleaning, tokenization, stopword removal, and stemming).
3. **Evaluation**: Achieved **~91.3% Accuracy** along with strong precision and recall.
4. **User Interface**: Developed a dynamic, responsive web interface using Streamlit to provide instant authenticity predictions and confidence scores.

## 🚀 Quick Start

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
The `train.py` script automatically downloads the dataset, cleans the text, extracts TF-IDF features, trains the model, and saves `model.pkl` and `vectorizer.pkl`.
*(Note: I have already run this for you! You do not need to run it again unless you want to retrain the model).*
```bash
python train.py
```

### 3. Run the Web Application
Launch the Streamlit interface to interact with the model:
```bash
streamlit run app.py
```
This will open a new tab in your default web browser where you can paste any news snippet and check its authenticity.

## 🧠 Technical Details
- **Dataset**: `fake_or_real_news.csv` (contains over 6,300 articles).
- **Text Vectorization**: `TfidfVectorizer` (TF-IDF) limited to the top 5,000 features.
- **Model**: `LogisticRegression` from `scikit-learn`.
- **UI Framework**: `Streamlit` with custom CSS for a premium design aesthetic.

Enjoy your new Fake News Detector!
