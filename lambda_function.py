import json
import os
import logging
from pathlib import Path

from nlp.preprocessor import detect_language
from nlp.similarity import CorpusMatcher
from nlp.sentiment import SentimentAnalyzer
from nlp.speech import SpeechProcessor
from nlp.corpus_loader import CorpusLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
SUPPORTED_LANGS = {'pt', 'en'}

corpus_text = Path('/var/task/data/corpus.txt').read_text()
matcher = CorpusMatcher(corpus_text)
sentiment_analyzer = SentimentAnalyzer()
speech_processor = SpeechProcessor()

_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from nlp.summarizer import TextSummarizer
        _summarizer = TextSummarizer()
    return _summarizer


def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))
        
        if 'message' not in body:
            return response(200, {'status': 'ok'})
        
        message = body['message']
        chat_id = message['chat']['id']
        
        if 'voice' in message:
            reply = process_voice(message)
        elif 'text' in message:
            reply = process_text(message['text'])
        else:
            reply = "Envie texto ou áudio."
        
        send_telegram_message(chat_id, reply)
        
        return response(200, {'status': 'processed'})
        
    except Exception as e:
        logger.error(f"Erro: {e}")
        return response(500, {'error': str(e)})


def process_text(text: str) -> str:
    lang = detect_language(text)
    
    if lang not in SUPPORTED_LANGS:
        return f"Idioma '{lang}' detectado. Suporto: {', '.join(SUPPORTED_LANGS)}."
    
    if text.startswith('/'):
        return handle_command(text, lang)
    
    sentiment = sentiment_analyzer.analyze(text)
    prefix = sentiment_analyzer.get_empathetic_prefix(sentiment['label'])
    
    if sentiment_analyzer.should_intervene(text):
        prefix = "Percebo que você pode estar passando por dificuldades. "
    
    best_match, score = matcher.find_best_match(text)
    
    if best_match:
        return f"{prefix}{best_match}"
    
    fallbacks = {
        'pt': "Não encontrei uma resposta adequada. Pode reformular?",
        'en': "I couldn't find a suitable answer. Can you rephrase?",
    }
    return prefix + fallbacks.get(lang, fallbacks['pt'])


def process_voice(message: dict) -> str:
    voice = message['voice']
    file_id = voice['file_id']
    
    file_url = get_telegram_file_url(file_id)
    
    if not file_url:
        return "Erro ao obter arquivo de áudio."
    
    text = speech_processor.process_telegram_voice(file_url)
    
    if text.startswith('['):  # Erro
        return text
    
    response_text = process_text(text)
    return f"🎤 Você disse: \"{text}\"\n\n{response_text}"


def handle_command(text: str, lang: str) -> str:
    cmd = text.split()[0].lower()
    args = text[len(cmd):].strip()
    
    commands = {
        '/start': lambda: get_welcome_message(lang),
        '/help': lambda: get_help_message(lang),
        '/lang': lambda: f"Idioma detectado: {lang}",
        '/sentiment': lambda: format_sentiment(sentiment_analyzer.analyze(args)),
        '/summarize': lambda: summarize_text(args),
        '/wiki': lambda: load_wiki_corpus(args, lang),
    }
    
    handler = commands.get(cmd)
    if handler:
        return handler()
    
    return "Comando não reconhecido. Use /help."


def format_sentiment(scores: dict) -> str:
    return (
        f"📊 Análise de Sentimento:\n"
        f"• Positivo: {scores['pos']:.1%}\n"
        f"• Neutro: {scores['neu']:.1%}\n"
        f"• Negativo: {scores['neg']:.1%}\n"
        f"• Classificação: {scores['label'].upper()}"
    )


def summarize_text(text: str) -> str:
    if len(text) < 100:
        return "Texto muito curto para resumir."
    
    summarizer = get_summarizer()
    summary = summarizer.summarize(text)
    return f"📝 Resumo:\n{summary}"


def load_wiki_corpus(topic: str, lang: str) -> str:
    global matcher
    try:
        text = CorpusLoader.from_wikipedia(topic, lang)
        matcher = CorpusMatcher(text)
        return f"✅ Corpus atualizado com artigo: {topic}"
    except Exception as e:
        return f"❌ Erro ao carregar Wikipedia: {e}"


def get_welcome_message(lang: str) -> str:
    messages = {
        'pt': "👋 Olá! Sou um bot com NLP. Envie texto ou áudio.",
        'en': "👋 Hello! I'm an NLP bot. Send text or audio.",
    }
    return messages.get(lang, messages['pt'])


def get_help_message(lang: str) -> str:
    return (
        "📖 Comandos disponíveis:\n"
        "/start - Iniciar\n"
        "/help - Esta mensagem\n"
        "/lang - Detectar idioma\n"
        "/sentiment <texto> - Analisar sentimento\n"
        "/summarize <texto> - Resumir texto\n"
        "/wiki <tópico> - Carregar Wikipedia como corpus\n\n"
        "💬 Ou simplesmente envie uma mensagem!"
    )


def get_telegram_file_url(file_id: str) -> str:
    import urllib.request
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile?file_id={file_id}"
    
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
            file_path = data['result']['file_path']
            return f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
    except:
        return None


def send_telegram_message(chat_id: int, text: str):
    import urllib.request
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = json.dumps({'chat_id': chat_id, 'text': text}).encode()
    
    req = urllib.request.Request(
        url, 
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        logger.error(f"Erro ao enviar mensagem: {e}")


def response(status_code: int, body: dict) -> dict:
    return {
        'statusCode': status_code,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(body)
    }
