import numpy as np
import sounddevice as sd
from app.core.config import SAMPLE_RATE


def capture_noise_profile(duration=2):
    print("Capturando perfil de ruído de fundo...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)


def remove_static_noise(audio, noise_profile):
    energy = lambda x: np.sqrt(np.mean(x ** 2))
    if energy(noise_profile) == 0:
        return audio
    noise_energy = energy(noise_profile)
    audio_energy = energy(audio)
    if audio_energy < 1.5 * noise_energy:
        return np.zeros_like(audio)
    return audio
