from transformers import pipeline
import os

os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'


class TextSummarizer:    
    def __init__(self, model: str = 'facebook/bart-large-cnn'):
        self.pipe = pipeline('summarization', model=model)
    
    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        if len(text.split()) < min_length:
            return text
        
        result = self.pipe(
            text, 
            max_length=max_length, 
            min_length=min_length,
            do_sample=False
        )
        return result[0]['summary_text']


class TextGenerator:    
    def __init__(self, model: str = 'gpt2'):
        self.pipe = pipeline('text-generation', model=model)
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        result = self.pipe(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
        return result[0]['generated_text']


class QuestionAnswering:
    def __init__(self, model: str = 'deepset/roberta-base-squad2'):
        self.pipe = pipeline('question-answering', model=model)
    
    def answer(self, question: str, context: str) -> dict:
        result = self.pipe(question=question, context=context)
        return {
            'answer': result['answer'],
            'score': result['score']
        }
