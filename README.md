# AI-Powered Fake News Detection System

This repository contains an end-to-end Machine Learning pipeline designed to identify and classify fake news articles. By leveraging Natural Language Processing (NLP) techniques and a Logistic Regression classifier, the system analyzes linguistic patterns to determine the authenticity of news text.

## Overview

The spread of misinformation is a critical challenge in the digital age. This project addresses the issue by providing a reliable, automated tool that evaluates news snippets in real-time. It features a custom-built NLP pipeline for text preprocessing, a trained machine learning model for classification, and a responsive web interface for user interaction.

## Key Features
- **Real-Time Analysis**: Instant classification of news articles into "Real" or "Fake".
- **Confidence Scoring**: Provides probability percentages to show the model's confidence in its predictions.
- **Robust NLP Pipeline**: Custom text cleaning, tokenization, stopword removal, and stemming using NLTK.
- **Interactive UI**: A modern, user-friendly dashboard built with Streamlit, featuring custom CSS.

## Project Structure
- `app.py`: The Streamlit web application script.
- `train.py`: The machine learning pipeline script for training the model.
- `fake_or_real_news.csv`: The dataset used for training and evaluation.
- `model.pkl`: The saved Logistic Regression classification model.
- `vectorizer.pkl`: The saved TF-IDF vectorizer.

## Quick Start

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
The `train.py` script automatically cleans the text, extracts TF-IDF features, trains the model, and saves `model.pkl` and `vectorizer.pkl`.
*(Note: Pre-trained models are already included in the repository. You only need to run this if you wish to retrain).*
```bash
python train.py
```

### 3. Run the Web Application
Launch the Streamlit interface to interact with the model:
```bash
streamlit run app.py
```
This will open a new tab in your default web browser where you can paste any news snippet and check its authenticity.

## Technical Details
- **Dataset**: `fake_or_real_news.csv` (contains over 6,300 articles with 'title', 'text', and 'label' columns).
- **Text Vectorization**: `TfidfVectorizer` (TF-IDF) limited to the top 5,000 features.
- **Model**: `LogisticRegression` from `scikit-learn`.
- **Performance**: The model achieves ~91.3% Accuracy on the testing set, demonstrating strong precision and recall.
- **UI Framework**: `Streamlit` with custom CSS for a premium design aesthetic.
