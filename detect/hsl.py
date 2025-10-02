# ultra_simple_detector.py
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Simple training data
real_news = ["Scientists make breakthrough in renewable energy", "Economy shows strong growth this quarter"]
fake_news = ["Aliens visit White House in secret meeting", "One simple trick to lose weight overnight"]

# Train model
vectorizer = TfidfVectorizer()
model = LogisticRegression()

# Prepare data
texts = real_news + fake_news
labels = ['real'] * len(real_news) + ['fake'] * len(fake_news)

X = vectorizer.fit_transform(texts)
model.fit(X, labels)

# Streamlit app
st.title("📰 Fake News Checker")
st.write("Paste any news below to check if it's REAL or FAKE")

news_input = st.text_area("News Article:", height=150)

if st.button("Check News"):
    if news_input:
        # Predict
        news_vector = vectorizer.transform([news_input])
        prediction = model.predict(news_vector)[0]
        probability = model.predict_proba(news_vector)[0]
        
        confidence = max(probability)
        
        if prediction == 'real':
            st.success(f"✅ REAL NEWS (Confidence: {confidence:.1%})")
        else:
            st.error(f"❌ FAKE NEWS (Confidence: {confidence:.1%})")
    else:
        st.warning("Please enter some news text!")