import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class DataPreprocessor:
    def __init__(self, category="Toys_and_Games", max_length=512):
        self.category = category
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def load_data(self):
        """Carica il dataset Amazon Reviews dalla categoria specificata"""
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", self.category)
        return dataset

    def preprocess_text(self, text):
        """Preprocessa il testo delle recensioni"""
        return self.tokenizer(text, 
                            truncation=True,
                            padding='max_length',
                            max_length=self.max_length,
                            return_tensors='pt')

    def get_sentiment(self, rating):
        """Converte il rating in stelle in un sentiment categorico"""
        if rating <= 2:
            return 0  # Negativo
        elif rating == 3:
            return 1  # Neutro
        else:
            return 2  # Positivo

    def prepare_dataset(self):
        """Prepara il dataset per il training"""
        dataset = self.load_data()
        
        # Estrai features e labels
        reviews = dataset['train']['review_body']
        ratings = dataset['train']['star_rating']
        
        # Converti ratings in sentiment labels
        sentiments = [self.get_sentiment(rating) for rating in ratings]
        
        # Tokenizza il testo
        encoded_reviews = [self.preprocess_text(review) for review in reviews]
        
        # Split train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            encoded_reviews, sentiments, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)