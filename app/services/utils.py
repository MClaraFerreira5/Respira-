import os
import soundfile as sf
from datetime import datetime


def save_event_audio(audio, sample_rate, label, count):
    """Salva o áudio da tog detectada"""

    os.makedirs(f"{label}_detections", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_detections/{label}_{timestamp}_{count}.wav"
    sf.write(filename, audio, sample_rate)
    print(f"Áudio de {label} salvo como {filename}")
