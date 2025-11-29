from goose3 import Goose
from pathlib import Path


class CorpusLoader:
    @staticmethod
    def from_url(url: str) -> str:
        g = Goose()
        article = g.extract(url=url)
        return article.cleaned_text
    
    @staticmethod
    def from_file(path: str) -> str:
        return Path(path).read_text(encoding='utf-8')
    
    @staticmethod
    def from_wikipedia(topic: str, lang: str = 'pt') -> str:
        url = f"https://{lang}.wikipedia.org/wiki/{topic}"
        return CorpusLoader.from_url(url)
