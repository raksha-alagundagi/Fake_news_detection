import os
import re
import pandas as pd
import numpy as np
import urllib.request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

DATA_URL = 'https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv'
DATA_FILE = 'fake_or_real_news.csv'

def download_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")
    else:
        print("Dataset already exists.")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove non-alphabet characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    words = [ps.stem(word) for word in words if not word in set(all_stopwords)]
    return ' '.join(words)

def train_model():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    print("Dataset shape:", df.shape)
    
    # We will use 'text' column for prediction. Let's merge 'title' and 'text'
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # Map label to binary: FAKE -> 1, REAL -> 0
    df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0})
    
    # Drop rows with NaN labels
    df = df.dropna(subset=['label'])
    
    print("Cleaning text (this may take a few minutes)...")
    # Apply text cleaning
    df['clean_content'] = df['content'].apply(clean_text)
    
    X = df['clean_content'].values
    y = df['label'].values
    
    print("Vectorizing data...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    print("Saving model and vectorizer...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Done! Artifacts saved as model.pkl and vectorizer.pkl.")

if __name__ == '__main__':
    download_data()
    train_model()
