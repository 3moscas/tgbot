from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .preprocessor import preprocess, tokenize_sentences, detect_language


class CorpusMatcher:
    def __init__(self, corpus_text: str):
        self.sentences = tokenize_sentences(corpus_text)
        self.vectorizer = TfidfVectorizer()
        
        self.processed_sentences = [
            preprocess(s, detect_language(s)) for s in self.sentences
        ]
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_sentences)
    
    def find_best_match(self, query: str, threshold: float = 0.1) -> tuple:
        lang = detect_language(query)
        processed_query = preprocess(query, lang)
        
        query_vector = self.vectorizer.transform([processed_query])
        
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= threshold:
            return self.sentences[best_idx], best_score
        
        return None, 0
    
    def find_top_matches(self, query: str, n: int = 3) -> list:
        lang = detect_language(query)
        processed_query = preprocess(query, lang)
        
        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(similarities)[-n:][::-1]
        
        return [(self.sentences[i], similarities[i]) for i in top_indices]
