from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from deep_translator import GoogleTranslator
import pickle
from pathlib import Path


class SentimentAnalyzer:    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.translator = GoogleTranslator(source='auto', target='en')
    
    def analyze(self, text: str) -> dict:
        try:
            translated = self.translator.translate(text)
        except:
            translated = text
        
        scores = self.vader.polarity_scores(translated)
        
        if scores['compound'] >= 0.05:
            label = 'positive'
        elif scores['compound'] <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        scores['label'] = label
        return scores
    
    def should_intervene(self, text: str, threshold: float = -0.3) -> bool:
        scores = self.analyze(text)
        return scores['compound'] <= threshold
    
    def get_empathetic_prefix(self, sentiment_label: str) -> str:
        prefixes = {
            'negative': "Entendo que pode estar passando por um momento difícil. ",
            'neutral': "",
            'positive': "Que bom! "
        }
        return prefixes.get(sentiment_label, "")


class TrainableSentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = DecisionTreeClassifier()
        self.is_trained = False
    
    def train(self, texts: list, labels: list):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def predict(self, text: str) -> int:
        if not self.is_trained:
            raise ValueError("Modelo não treinado")
        
        X = self.vectorizer.transform([text])
        return self.classifier.predict(X)[0]
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
            self.is_trained = True
