import soundfile as sf
import speech_recognition as sr
from core.config import KEYWORD


def recognize_keyword(audio, sample_rate):
    try:
        filename = "temp_speech.wav"
        sf.write(filename, audio, sample_rate)
        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data, language="pt-BR")
        print(f"Transcrição: {transcript}")
        return KEYWORD.lower() in transcript.lower()
    except Exception as e:
        print(f"Erro no reconhecimento de fala: {e}")
        return False
