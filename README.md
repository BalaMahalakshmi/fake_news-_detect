# fake_news-_detect
# 📰 Fake News Detection Project

## 🧠 Project Overview

In today's information era, the spread of **fake news** through digital platforms has become a serious problem. This project uses **Natural Language Processing (NLP)** and **machine learning** to detect whether a news article is **real or fake**.

---

## 🎯 Objective

To develop a binary classification model that can:
- Identify fake news articles based on textual content (e.g., headlines or full body).
- Use NLP techniques like **TF-IDF** and models like **Logistic Regression** or **Naive Bayes**.
- Evaluate model performance using **accuracy**, **precision**, **recall**, and **F1-score**.

---

## 🧰 Tools & Technologies

| Tool | Description |
|------|-------------|
| Python | Programming language |
| Pandas | Data loading and preprocessing |
| scikit-learn | ML model building and evaluation |
| NLTK / re | Text preprocessing |
| TF-IDF | Feature extraction from text |
| Matplotlib / Seaborn | Visualizations |

---

## 📁 Dataset

**Dataset Used**: [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

**Files:**
- `True.csv` – Real news
- `Fake.csv` – Fake news

**Columns Used:**
- `title` – News headline
- `text` – Full article text
- `label` – Real (1) or Fake (0)

---

## 🚀 Project Workflow

1. **Load Data**: Combine `True.csv` and `Fake.csv` into one labeled dataset.
2. **Preprocess Text**:
   - Lowercasing, punctuation removal, stopword removal
   - Tokenization (optional), stemming/lemmatization
3. **Feature Engineering**:
   - Use **TF-IDF Vectorizer** to convert text into numerical features
4. **Model Training**:
   - Train using **Logistic Regression** or **Naive Bayes**
5. **Evaluation**:
   - Accuracy, precision, recall, F1-score
   - Confusion matrix and classification report
6. **Visualization**:
   - Word cloud for fake vs real headlines
   - Bar chart of top predictive words (optional)

---

## 📊 Sample Results

| Metric | Score |
|--------|-------|
| Accuracy | 95.4% |
| Precision | 94.7% |
| Recall | 96.1% |
| F1-Score | 95.3% |

> *(May vary based on preprocessing and model type)*

---

## 📌 How to Run

1. **Clone this repo**
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
