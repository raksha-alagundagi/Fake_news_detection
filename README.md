# AI-Powered Fake News Detection System

## Problem Statement
The spread of misinformation and fake news is a critical challenge in the digital age, often manipulating public perception and causing social harm. Manually verifying the authenticity of every news article is nearly impossible due to the sheer volume of information generated daily. This project addresses this issue by providing a reliable, automated tool that evaluates news snippets in real-time, helping users distinguish between genuine reporting and fabricated stories using machine learning.

## Features
- **Real-Time Analysis**: Instant classification of news articles into "Real" or "Fake".
- **Confidence Scoring**: Provides probability percentages to show the model's confidence in its predictions.
- **Robust NLP Pipeline**: Custom text cleaning, tokenization, stopword removal, and stemming.
- **Interactive UI**: A modern, user-friendly dashboard built with Streamlit, featuring custom CSS.
- **High Accuracy**: The model achieves ~91.3% accuracy on testing data.

## Tech Stack
- **Language**: Python 3.8+
- **Machine Learning**: Scikit-Learn (Logistic Regression)
- **NLP**: NLTK (Natural Language Toolkit), TF-IDF Vectorizer
- **Data Manipulation**: Pandas, NumPy
- **Frontend / UI**: Streamlit, HTML, CSS

## Screenshots / Output
*(Add your <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a120e2fc-77a1-496e-8a13-19d78bc152fe" />
project screenshot here)*
![App Screenshot](screenshot.png)

## How to Run

### 1. Install Dependencies
Ensure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
Pre-trained models (`model.pkl`, `vectorizer.pkl`) are already included. If you want to retrain the model on the dataset (`fake_or_real_news.csv`), run:
```bash
python train.py
```

### 3. Run the Web Application
Launch the Streamlit interface to interact with the model:
```bash
streamlit run app.py
```
This will open a new tab in your default web browser where you can paste any news snippet and check its authenticity.

## Future Improvements
- **Advanced Models**: Implement deep learning architectures like LSTM or BERT for contextual understanding.
- **Web Scraping Integration**: Allow users to paste a URL instead of text to automatically extract and verify the article.
- **Source Verification**: Integrate APIs to check the credibility of the publisher or domain.
- **Multilingual Support**: Extend the NLP pipeline to detect fake news in languages other than English.
