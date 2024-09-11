import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # TODO: Implement data preprocessing logic
    pass

def split_data(df, test_size=0.2):
    return train_test_split(df, test_size=test_size, random_state=42)
