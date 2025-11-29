import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect, LangDetectException

STOPWORDS = {
    'pt': set(stopwords.words('portuguese')),
    'en': set(stopwords.words('english')),
}


def detect_language(text: str) -> str:
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    
    try:
        lang = detect(text)
        return lang if lang in ['pt', 'en'] else 'pt'
    except LangDetectException:
        return 'pt'


def preprocess(text: str, lang: str = 'pt') -> str:
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove menções (@usuario)
    text = re.sub(r'@\w+', '', text)
    
    # Remove números
    text = re.sub(r'\d+', '', text)
    
    # Remove pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    stop_words = STOPWORDS.get(lang, STOPWORDS['pt'])
    tokens = word_tokenize(text, language='portuguese' if lang == 'pt' else 'english')
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    
    return ' '.join(tokens)


def tokenize_sentences(text: str) -> list:
    return sent_tokenize(text)
