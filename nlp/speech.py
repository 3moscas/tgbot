import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os
import urllib.request


class SpeechProcessor:
    LANG_MAP = {
        'pt': 'pt-BR',
        'en': 'en-US',
        'es': 'es-ES',
    }
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def download_telegram_audio(self, file_url: str) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.ogg')
        urllib.request.urlretrieve(file_url, temp_file.name)
        return temp_file.name
    
    def convert_ogg_to_wav(self, ogg_path: str) -> str:
        wav_path = ogg_path.replace('.ogg', '.wav')
        audio = AudioSegment.from_ogg(ogg_path)
        audio.export(wav_path, format='wav')
        return wav_path
    
    def transcribe(self, audio_path: str, language: str = 'pt') -> str:
        lang_code = self.LANG_MAP.get(language, 'pt-BR')
        
        try:
            if audio_path.endswith('.ogg'):
                audio_path = self.convert_ogg_to_wav(audio_path)
            
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(
                audio_data, 
                language=lang_code
            )
            return text
            
        except sr.UnknownValueError:
            return "[Não foi possível entender o áudio]"
        except sr.RequestError as e:
            return f"[Erro no serviço de reconhecimento: {e}]"
        finally:
            self._cleanup(audio_path)
    
    def process_telegram_voice(self, file_url: str, language: str = 'pt') -> str:
        ogg_path = self.download_telegram_audio(file_url)
        return self.transcribe(ogg_path, language)
    
    def _cleanup(self, path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
            wav_path = path.replace('.ogg', '.wav')
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except:
            pass
