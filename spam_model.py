import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import joblib
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class SpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
        
    def train(self, data_path):
        # Read the dataset with explicit encoding
        df = pd.read_csv(data_path, encoding='latin-1')
        
        # Preprocess the text
        df['processed_text'] = df['v2'].apply(self.preprocess_text)
        
        # Convert labels
        df['label'] = (df['v1'] == 'spam').astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Vectorize the text
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_vectorized, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vectorized)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict(self, text):
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([processed_text])
        # Make prediction
        prediction = self.model.predict(text_vectorized)[0]
        # Get probability scores
        proba = self.model.predict_proba(text_vectorized)[0]
        return prediction, proba
    
    def save_model(self, model_path='spam_model.joblib', vectorizer_path='vectorizer.joblib'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
    
    def load_model(self, model_path='spam_model.joblib', vectorizer_path='vectorizer.joblib'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

if __name__ == "__main__":
    # Initialize and train the model
    detector = SpamDetector()
    detector.train("spam.csv")
    # Save the model
    detector.save_model() 