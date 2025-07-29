import numpy as np
from scipy import signal
from core.config import SAMPLE_RATE


def preprocess_audio(audio):
    """Melhoria no pré-processamento do áudio"""

    # Normalização
    audio = audio / (np.max(np.abs(audio)) + 1e-8)  # Evita divisão por zero

    # Filtro passa-banda
    lowcut = 50
    highcut = 3000
    nyquist = 0.5 * SAMPLE_RATE
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    audio = signal.filtfilt(b, a, audio)

    # Remover Silêncio
    energy = np.sqrt(np.mean(audio ** 2))
    threshold = energy * 0.05  # Ajuste conforme necessário
    valid_samples = np.where(np.abs(audio) > threshold)[0]

    if len(valid_samples) > 0:
        start = max(0, valid_samples[0] - 100)
        end = min(len(audio), valid_samples[-1] + 100)
        audio[start:end]

    return audio.astype(np.float32)
