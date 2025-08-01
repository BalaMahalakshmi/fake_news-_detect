# Fake News Detection System
# Complete implementation with preprocessing, training, evaluation, and interface

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Comprehensive text preprocessing"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, file_path=None):
        """Load and preprocess the dataset"""
        # If no file path provided, create sample data
        if file_path is None:
            print("Creating sample dataset for demonstration...")
            sample_data = self.create_sample_data()
            df = pd.DataFrame(sample_data)
        else:
            print(f"Loading data from {file_path}...")
            df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Handle different column names that might exist in Kaggle datasets
        if 'title' in df.columns and 'text' in df.columns:
            df['content'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
        elif 'title' in df.columns:
            df['content'] = df['title'].astype(str)
        elif 'text' in df.columns:
            df['content'] = df['text'].astype(str)
        elif 'content' not in df.columns:
            raise ValueError("Dataset must contain 'title', 'text', or 'content' column")
        
        # Handle different label column names
        if 'label' not in df.columns:
            if 'fake' in df.columns:
                df['label'] = df['fake']
            elif 'is_fake' in df.columns:
                df['label'] = df['is_fake']
            elif 'target' in df.columns:
                df['label'] = df['target']
            else:
                raise ValueError("Dataset must contain a label column")
        
        # Convert labels to binary (0 for real, 1 for fake)
        if df['label'].dtype == 'object':
            df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1, 'real': 0, 'fake': 1, 
                                         'True': 1, 'False': 0, True: 1, False: 0})
        
        # Remove rows with missing content or labels
        df = df.dropna(subset=['content', 'label'])
        
        print("Preprocessing text data...")
        df['processed_content'] = df['content'].apply(self.preprocess_text)
        
        # Remove empty processed content
        df = df[df['processed_content'].str.len() > 0]
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        sample_data = {
            'content': [
                "Scientists discover new breakthrough in renewable energy technology",
                "SHOCKING: Aliens landed in New York City yesterday, government covers up!",
                "Local community comes together to support flood victims",
                "UNBELIEVABLE: Eating this one food cures all diseases instantly!",
                "New study shows benefits of regular exercise for mental health",
                "BREAKING: Celebrity caught in scandal that will shock you!",
                "Environmental protection agency announces new climate initiatives",
                "MIRACLE CURE: Doctors hate this one simple trick!",
                "Education funding increases in local school districts",
                "EXPOSED: Government hiding truth about moon landing hoax!",
                "Research shows promise in new cancer treatment approach",
                "CLICK HERE: You won't believe what happens next!",
                "Community garden project helps feed local families",
                "AMAZING: This simple method will make you rich overnight!",
                "Weather service issues flood warning for coastal areas"
            ],
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 0 = real, 1 = fake
        }
        return sample_data
    
    def train_model(self, df):
        """Train the fake news detection model"""
        print("Training the model...")
        
        X = df['processed_content']
        y = df['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline with TF-IDF and Logistic Regression
        pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Store for later use
        self.pipeline = pipeline
        self.is_trained = True
        
        return X_train, X_test, y_train, y_test, y_pred, y_pred_proba
    
    def evaluate_model(self, y_test, y_pred, y_pred_proba=None):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"True Negatives (Real predicted as Real): {cm[0,0]}")
        print(f"False Positives (Real predicted as Fake): {cm[0,1]}")
        print(f"False Negatives (Fake predicted as Real): {cm[1,0]}")
        print(f"True Positives (Fake predicted as Fake): {cm[1,1]}")
        
        # Analyze false positives and negatives
        self.analyze_errors(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def analyze_errors(self, y_test, y_pred):
        """Analyze false positives and false negatives"""
        print("\n" + "="*50)
        print("ERROR ANALYSIS")
        print("="*50)
        
        # Get indices of false positives and false negatives
        fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
        fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
        
        print(f"\nFalse Positives (Real news classified as Fake): {len(fp_indices)}")
        print(f"False Negatives (Fake news classified as Real): {len(fn_indices)}")
        
        if len(fp_indices) > 0:
            print(f"\nFalse Positive Rate: {len(fp_indices) / len(y_test[y_test == 0]):.4f}")
        
        if len(fn_indices) > 0:
            print(f"False Negative Rate: {len(fn_indices) / len(y_test[y_test == 1]):.4f}")
    
    def predict_single_article(self, text):
        """Predict if a single article is fake or real"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        processed_text = self.preprocess_text(text)
        prediction = self.pipeline.predict([processed_text])[0]
        probability = self.pipeline.predict_proba([processed_text])[0]
        
        result = {
            'text': text[:200] + "..." if len(text) > 200 else text,
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': max(probability),
            'fake_probability': probability[1],
            'real_probability': probability[0]
        }
        
        return result
    
    def create_simple_interface(self):
        """Simple command-line interface for testing"""
        print("\n" + "="*60)
        print("FAKE NEWS DETECTION INTERFACE")
        print("="*60)
        print("Enter news text to analyze (type 'quit' to exit)")
        print("-" * 60)
        
        while True:
            user_input = input("\nEnter news text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            try:
                result = self.predict_single_article(user_input)
                
                print(f"\nPREDICTION RESULTS:")
                print(f"Text: {result['text']}")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Real Probability: {result['real_probability']:.4f}")
                print(f"Fake Probability: {result['fake_probability']:.4f}")
                
                if result['prediction'] == 'FAKE':
                    print("⚠️  This appears to be FAKE news!")
                else:
                    print("✅ This appears to be REAL news.")
                    
            except Exception as e:
                print(f"Error processing text: {e}")

def main():
    """Main execution function"""
    print("🔍 FAKE NEWS DETECTION SYSTEM")
    print("="*50)
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Load and preprocess data
    # For this demo, we'll use sample data. In practice, use:
    # df = detector.load_and_preprocess_data('path/to/your/dataset.csv')
    df = detector.load_and_preprocess_data()
    
    # Train model
    X_train, X_test, y_train, y_test, y_pred, y_pred_proba = detector.train_model(df)
    
    # Evaluate model
    metrics = detector.evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Test with sample articles
    print("\n" + "="*50)
    print("TESTING WITH SAMPLE ARTICLES")
    print("="*50)
    
    test_articles = [
        "Scientists at MIT have developed a revolutionary new solar panel technology that could change renewable energy forever.",
        "BREAKING: Government officials caught hiding alien technology in secret underground facility!",
        "Local school district announces new funding for arts and music programs.",
        "SHOCKING: This one weird trick will make you lose 50 pounds in a week!"
    ]
    
    for i, article in enumerate(test_articles, 1):
        print(f"\nTest Article {i}:")
        result = detector.predict_single_article(article)
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")
    
    # Launch interactive interface
    print("\n" + "="*50)
    print("INTERACTIVE TESTING")
    print("="*50)
    
    choice = input("Would you like to test the interactive interface? (y/n): ").lower()
    if choice in ['y', 'yes']:
        detector.create_simple_interface()
    
    print("\n✅ Fake News Detection System Demo Complete!")

if __name__ == "__main__":
    main()