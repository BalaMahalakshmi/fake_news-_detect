# Fake News Detection System
# Complete implementation with preprocessing, training, evaluation, and interface

import pandas as pd
import numpy as np
import re

# NLTK imports with error handling
try:
    import nltk
    # Ensure NLTK data is available
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        nltk_available = True
        # print(" NLTK loaded successfully")
    except LookupError:
        print("⚠️ NLTK data not found. Downloading...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        nltk_available = True
except ImportError:
    print(" NLTK not installed. Install with: pip install nltk")
    nltk_available = False

# Scikit-learn imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    sklearn_available = True
    # print("Scikit-learn loaded successfully")
except ImportError:
    print(" Scikit-learn not installed. Install with: pip install scikit-learn")
    sklearn_available = False

class FakeNewsDetector:
    def _init_(self):
        self.vectorizer = None
        self.model = None
        self.stop_words = None
        
        # Initialize stopwords if NLTK is available
        if nltk_available:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                print("Using basic stopwords list")
                self.stop_words = set(['the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are'])
        else:
            self.stop_words = set(['the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are'])
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Tokenization and stopword removal
        if nltk_available:
            try:
                tokens = word_tokenize(text)
                tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
                return ' '.join(tokens)
            except:
                # Fallback to simple tokenization
                words = text.split()
                words = [word for word in words if word not in self.stop_words and len(word) > 2]
                return ' '.join(words)
        else:
            # Simple tokenization without NLTK
            words = text.split()
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            return ' '.join(words)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Creating sample dataset for demonstration...")
        
        # Sample data for demonstration
        sample_data = {
            'content': [
                "Scientists discover breakthrough in renewable energy technology",
                "SHOCKING: Government hides alien contact for decades!!!",
                "Local community comes together to help flood victims",
                "You won't believe what happens next! Doctors hate this trick!",
                "Research shows promising results for new cancer treatment",
                "BREAKING: Celebrity secretly controls world economy",
                "University announces new scholarship program for students",
                "Miracle cure discovered! Big pharma doesn't want you to know!",
                "City council approves new public transportation funding",
                "EXCLUSIVE: Moon landing was definitely fake, insider reveals",
                "Environmental group plants 10,000 trees in local park",
                "Secret government mind control through 5G towers exposed!",
                "School district implements new literacy program",
                "One weird trick eliminates belly fat overnight!",
                "Medical journal publishes peer-reviewed vaccine study"
            ],
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 0=Real, 1=Fake
        }
        
        df = pd.DataFrame(sample_data)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Preprocess text data
        print("Preprocessing text data...")
        df['processed_content'] = df['content'].apply(self.preprocess_text)
        
        # Remove empty processed content
        df = df[df['processed_content'].str.len() > 0]
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def train_model(self, df):
        """Train the fake news detection model"""
        if not sklearn_available:
            print("Cannot train model without scikit-learn")
            return False
        
        print("Training fake news detection model...")
        
        # Prepare features and labels
        X = df['processed_content']
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Vectorization
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        return True
    
    def predict(self, text):
        """Predict if news is fake or real"""
        if self.model is None or self.vectorizer is None:
            return "Model not trained yet!"
        
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return "Please provide valid text"
        
        # Vectorize and predict
        text_vec = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        result = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        confidence = max(probability) * 100
        
        return f"Prediction: {result} (Confidence: {confidence:.1f}%)"

def main():
    """Main function to run the fake news detector"""
    print("🔍 FAKE NEWS DETECTION SYSTEM")
    print("=" * 50)
    
    # Check dependencies
    if not sklearn_available:
        print(" Please install scikit-learn: pip install scikit-learn")
        return
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Load and preprocess data
    try:
        df = detector.load_and_preprocess_data()
        
        # Train model
        if detector.train_model(df):
            print("\n" + "="*50)
            print("🎉 Model trained successfully!")
            print("="*50)
            
            # Interactive prediction
            print("\n🔮 Test the model with your own text:")
            print("(Type 'quit' to exit)")
            
            while True:
                user_input = input("\nEnter news text: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Thanks for using Fake News Detector! 👋")
                    break
                
                if user_input:
                    result = detector.predict(user_input)
                    print(f"📊 {result}")
                else:
                    print("Please enter some text!")
        
    except Exception as e:
        print(f" Error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install pandas numpy scikit-learn nltk")

if __name__ == "_main_":
    main()