import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download NLTK resources (run once)
try:
    nltk.download('stopwords')
    nltk.download('punkt')
except:
    print("NLTK downloads may require manual installation")

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=52)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.le = LabelEncoder()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Clean and preprocess text data
        """
        if isinstance(text, float):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def load_and_prepare_data(self, file_path=None):
        """
        Load and prepare the dataset
        If no file path is provided, creates sample data for demonstration
        """
        if file_path:
            # Load from CSV file
            df = pd.read_csv(file_path)
        else:
            # Create sample data for demonstration
            print("No file provided. Creating sample data...")
            df = self.create_sample_data()
        
        # Display dataset info
        print("Dataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nClass Distribution:")
        print(df['label'].value_counts())
        
        return df
    
    def create_sample_data(self):
        """
        Create sample fake and real news data for demonstration
        """
        real_news = [
            "Scientists discover new species in Amazon rainforest with potential medical benefits",
            "Global leaders agree on climate change action plan at international summit",
            "Stock market shows positive growth amid economic recovery signals",
            "New education policy focuses on digital literacy and skill development",
            "Healthcare breakthrough: New treatment shows promise for rare diseases",
            "Renewable energy adoption reaches record levels worldwide",
            "Space agency launches satellite to monitor environmental changes",
            "Economic indicators suggest stable growth in coming quarters",
            "International collaboration leads to successful peace negotiations",
            "Technology company announces innovation in sustainable computing"
        ]
        
        fake_news = [
            "Aliens spotted in downtown area, government covering up evidence",
            "Secret cure for cancer discovered but hidden by pharmaceutical companies",
            "Celebrity death hoax spreads rapidly on social media platforms",
            "False weather predictions cause unnecessary panic among residents",
            "Conspiracy theory about moon landing gains traction online",
            "Fake product endorsement scams target vulnerable consumers",
            "Misleading health advice causes harm to followers",
            "Fabricated political scandal influences public opinion",
            "Bogus investment scheme promises unrealistic returns",
            "False historical claims rewritten in popular online forums"
        ]
        
        data = {
            'text': real_news + fake_news,
            'label': ['real'] * len(real_news) + ['fake'] * len(fake_news)
        }
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df, text_column='text', label_column='label'):
        """
        Prepare features and labels for training
        """
        # Preprocess text
        print("Preprocessing text data...")
        df['cleaned_text'] = df[text_column].apply(self.preprocess_text)
        
        # Prepare features using TF-IDF
        print("Creating TF-IDF features...")
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        
        # Prepare labels
        y = self.le.fit_transform(df[label_column])
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """
        Train both Random Forest and KNN models
        """
        print("Training Random Forest...")
        self.rf_classifier.fit(X_train, y_train)
        
        print("Training KNN...")
        self.knn_classifier.fit(X_train, y_train)
        
        print("Training completed!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate both models and return results
        """
        # Predictions
        rf_pred = self.rf_classifier.predict(X_test)
        knn_pred = self.knn_classifier.predict(X_test)
        
        # Calculate accuracies
        rf_accuracy = accuracy_score(y_test, rf_pred)
        knn_accuracy = accuracy_score(y_test, knn_pred)
        
        # Classification reports
        rf_report = classification_report(y_test, rf_pred, target_names=self.le.classes_)
        knn_report = classification_report(y_test, knn_pred, target_names=self.le.classes_)
        
        return {
            'rf_accuracy': rf_accuracy,
            'knn_accuracy': knn_accuracy,
            'rf_predictions': rf_pred,
            'knn_predictions': knn_pred,
            'rf_report': rf_report,
            'knn_report': knn_report
        }
    
    def plot_results(self, results, y_test):
        """
        Plot comparison of model performances
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy comparison
        models = ['Random Forest', 'KNN']
        accuracies = [results['rf_accuracy'], results['knn_accuracy']]
        
        axes[0].bar(models, accuracies, color=['skyblue', 'lightcoral'])
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        
        # Add accuracy values on bars
        for i, v in enumerate(accuracies):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Confusion matrix for Random Forest
        cm_rf = confusion_matrix(y_test, results['rf_predictions'])
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.le.classes_, yticklabels=self.le.classes_, ax=axes[1])
        axes[1].set_title('Random Forest - Confusion Matrix')
        
        # Confusion matrix for KNN
        cm_knn = confusion_matrix(y_test, results['knn_predictions'])
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Reds', 
                   xticklabels=self.le.classes_, yticklabels=self.le.classes_, ax=axes[2])
        axes[2].set_title('KNN - Confusion Matrix')
        
        plt.tight_layout()
        plt.show()
    
    def predict_news(self, text):
        """
        Predict whether a given text is fake or real news
        """
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)
        
        # Transform using the fitted vectorizer
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Get predictions from both models
        rf_pred = self.rf_classifier.predict(text_vector)
        knn_pred = self.knn_classifier.predict(text_vector)
        
        # Convert back to original labels
        rf_label = self.le.inverse_transform(rf_pred)[0]
        knn_label = self.le.inverse_transform(knn_pred)[0]
        
        # Get prediction probabilities
        rf_proba = self.rf_classifier.predict_proba(text_vector)[0]
        knn_proba = self.knn_classifier.predict_proba(text_vector)[0]
        
        return {
            'text': text,
            'random_forest': {
                'prediction': rf_label,
                'confidence': max(rf_proba)
            },
            'knn': {
                'prediction': knn_label,
                'confidence': max(knn_proba)
            }
        }

def main():
    """
    Main function to run the fake news detection system
    """
    # Initialize the detector
    detector = FakeNewsDetector()
    
    # Load data (replace with your dataset path)
    # For example: df = detector.load_and_prepare_data('fake_news_dataset.csv')
    df = detector.load_and_prepare_data()
    
    # Prepare features and labels
    X, y = detector.prepare_features(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Train models
    detector.train_models(X_train, y_train)
    
    # Evaluate models
    results = detector.evaluate_models(X_test, y_test)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE RESULTS")
    print("="*50)
    print(f"Random Forest Accuracy: {results['rf_accuracy']:.4f}")
    print(f"KNN Accuracy: {results['knn_accuracy']:.4f}")
    
    print("\nRANDOM FOREST CLASSIFICATION REPORT:")
    print(results['rf_report'])
    
    print("KNN CLASSIFICATION REPORT:")
    print(results['knn_report'])
    
    # Plot results
    detector.plot_results(results, y_test)
    
    # Test with custom examples
    print("\n" + "="*50)
    print("CUSTOM PREDICTION EXAMPLES")
    print("="*50)
    
    test_examples = [
        "Breaking: Scientists confirm major breakthrough in renewable energy technology that will change the world",
        "Shocking: Celebrities involved in secret alien worship ceremony exposed",
        "Government announces new economic stimulus package to support small businesses",
        "Viral hoax about miracle weight loss pill causes health concerns"
    ]
    
    for example in test_examples:
        prediction = detector.predict_news(example)
        print(f"\nText: {example}")
        print(f"Random Forest: {prediction['random_forest']['prediction']} "
              f"(Confidence: {prediction['random_forest']['confidence']:.3f})")
        print(f"KNN: {prediction['knn']['prediction']} "
              f"(Confidence: {prediction['knn']['confidence']:.3f})")
        print("-" * 80)

if __name__ == "__main__":
    main()