import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text):
    # convert to string
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def extract_features(df):
    # Preprocess the instruction text
    df['instruction'] = df['instruction'].astype(str)
    df['processed_instruction'] = df['instruction'].apply(preprocess_text)
    
    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_features = tfidf.fit_transform(df['processed_instruction'])
    
    # Length features
    df['instruction_length'] = df['instruction'].apply(len)
    df['word_count'] = df['instruction'].apply(lambda x: len(x.split()))
    
    # Combine all features
    feature_matrix = np.hstack((
        tfidf_features.toarray(),
        df[['instruction_length', 'word_count']].values
    ))
    
    return feature_matrix, df['category']

def preprocess_data(df):
    # Remove rows with non string instruction or category
    df = df[df['instruction'].apply(lambda x: isinstance(x, str))]
    df = df[df['category'].apply(lambda x: isinstance(x, str))]
    
    # Extract features and labels
    features, labels = extract_features(df)
    return features, labels

def split_data(features, labels, test_size=0.2):
    return train_test_split(features, labels, test_size=test_size, random_state=42)
