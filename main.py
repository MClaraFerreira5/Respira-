import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time
from datetime import datetime
from scipy import signal
import speech_recognition as sr

# Configura o FFmpeg para o PyDub
os.environ["PATH"] += os.pathsep + 'C:\\ffmpeg\\bin'

# Configurações
MODEL_URL = 'https://tfhub.dev/google/yamnet/1'
SAMPLE_RATE = 16000  # YAMNet requer áudio em 16kHz
RECORD_DURATION = 3  # Duração da gravação em segundos
THRESHOLDS = {        # Modificação: thresholds específicos por classe
    'Cough': 0.15,
    'Snore': 0.15,
    'Breathing': 0.15,
    'Sneeze': 0.15,
    'Gasp': 0.15,
    'Speech': 0.15
}

MIN_COUGH_DURATION = 0.2  # Duração mínima de uma tosse em segundos
COUGH_PATTERN_COUNT = 3  # Número mínimo de tosses para caracterizar um padrão
KEYWORD = "ajuda"

# Carrega o modelo YAMNet
model = hub.load(MODEL_URL)

# Índices das classes no YAMNet
CLASS_INDICES = {
    'Cough': 42,
    'Snore': 38,
    'Breathing': 36,
    'Speech': 65,
    'Silence': 1,
    'Sneeze': 44,
    'Sniff': 45,
    'Gasp': 39
}

class AudioHistory:
    def __init__(self, max_length=10):
        self.max_length = max_length
        self.predictions = []
        self.audio_segments = []
        self.timestamps = []

    def add_prediction(self, prediction, audio_segment=None):
        self.predictions.append(prediction)
        self.audio_segments.append(audio_segment)
        self.timestamps.append(datetime.now())

        if len(self.predictions) > self.max_length:
            self.predictions.pop(0)
            self.audio_segments.pop(0)
            self.timestamps.pop(0)
    
    def smooth(self):     # Modificação: suavização de predições
        if not self.predictions:
            return np.zeros(521)
        return np.mean(self.predictions, axis=0)

    def get_recent_pattern(self, window_size=3):
        if len(self.predictions) >= window_size:
            return self.predictions[-window_size:]
        return None

    def get_cough_count(self, threshold=0.3):  
        return sum(1 for p in self.predictions
                   if p[CLASS_INDICES['Cough']] > threshold)

def capture_noise_profile(duration=2):    # Modificação: captura de ruído ambiente
    print("Capturando perfil de ruído de fundo...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def remove_static_noise(audio, noise_profile):   
    energy = lambda x: np.sqrt(np.mean(x**2))
    if energy(noise_profile) == 0:
        return audio
    noise_energy = energy(noise_profile)
    audio_energy = energy(audio)
    if audio_energy < 1.5 * noise_energy:
        return np.zeros_like(audio)
    return audio

def preprocess_audio(audio):
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    lowcut = 50
    highcut = 3000
    nyquist = 0.5 * SAMPLE_RATE
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    audio = signal.filtfilt(b, a, audio)

    energy = np.sqrt(np.mean(audio ** 2))
    threshold = energy * 0.05
    valid_samples = np.where(np.abs(audio) > threshold)[0]

    if len(valid_samples) > 0:
        start = max(0, valid_samples[0] - 100)
        end = min(len(audio), valid_samples[-1] + 100)
        audio = audio[start:end]

    return audio.astype(np.float32)

def record_audio(duration=RECORD_DURATION, sample_rate=SAMPLE_RATE):
    try:
        print(f"\nGravando por {duration} segundos...")
        audio = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='float32')
        sd.wait()
        return np.squeeze(audio)
    except Exception as e:
        print(f"Erro na gravação: {e}")
        return np.zeros(int(duration * sample_rate))

def generate_spectrogram(audio, sample_rate=SAMPLE_RATE, prediction=None):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    spec, freqs, times, im = plt.specgram(audio, Fs=sample_rate,
                                          cmap=cm.viridis, NFFT=512,
                                          noverlap=256)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma do Áudio')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência (Hz)')

    plt.subplot(3, 1, 2)
    time_axis = np.linspace(0, len(audio) / sample_rate, num=len(audio))
    plt.plot(time_axis, audio)
    plt.title('Forma de Onda')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    if prediction is not None:
        plt.subplot(3, 1, 3)
        classes = list(CLASS_INDICES.keys())
        values = [prediction[idx] for idx in CLASS_INDICES.values()]
        colors = ['red' if cls == 'Cough' else 'blue' for cls in classes]
        bars = plt.bar(classes, values, color=colors)
        plt.title('Probabilidades das Classes')
        plt.ylabel('Probabilidade')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def predict_sound(audio):
    try:
        waveform = audio / (np.max(np.abs(audio)) + 1e-8)
        waveform = waveform.astype(np.float32)
        scores, embeddings, spectrogram = model(waveform)
        prediction = np.mean(scores, axis=0)
        top_class = np.argmax(prediction)
        confidence = prediction[top_class]
        return top_class, confidence, prediction
    except Exception as e:
        print(f"Erro na predição: {e}")
        return -1, 0, np.zeros(521)

def is_cough(prediction, history=None):
    cough_score = prediction[CLASS_INDICES['Cough']]
    silence_score = prediction[CLASS_INDICES['Silence']]
    above_threshold = cough_score > THRESHOLDS['Cough'] 
    not_silence = silence_score < 0.1
    is_pattern = False
    if history is not None and len(history.predictions) >= COUGH_PATTERN_COUNT:
        recent_coughs = sum(1 for p in history.predictions[-COUGH_PATTERN_COUNT:]
                            if p[CLASS_INDICES['Cough']] > THRESHOLDS['Cough'])  # Fixed: Use specific threshold
        is_pattern = recent_coughs >= COUGH_PATTERN_COUNT - 1
    return above_threshold and not_silence and (is_pattern or cough_score > 0.5)

def is_sneeze(prediction):
    return prediction[CLASS_INDICES['Sneeze']] > THRESHOLDS['Sneeze']  # Fixed: Use specific threshold

def is_snore(prediction):
    return prediction[CLASS_INDICES['Snore']] > THRESHOLDS['Snore']  # Fixed: Use specific threshold

def is_gasp(prediction):
    return prediction[CLASS_INDICES['Gasp']] > THRESHOLDS['Gasp']  # Fixed: Use specific threshold

def recognize_keyword(audio, sample_rate):
    """Realiza reconhecimento de fala e verifica se palavra-chave está presente"""
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

def save_event_audio(audio, sample_rate, label, count):
    os.makedirs(f"{label}_detections", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_detections/{label}_{timestamp}_{count}.wav"
    sf.write(filename, audio, sample_rate)
    print(f"Áudio de {label} salvo como {filename}")

def main():
    print("\n=== Sistema Avançado de Detecção de Tosse ===")
    print("Pressione Ctrl+C para sair\n")

    history = AudioHistory(max_length=10)
  #cough_count = 0
  #sneeze_count = 0
  #snore_count = 0
  #gasp_count = 0
    session_start = datetime.now()

    noise_profile = capture_noise_profile()  
    event_counts = {label: 0 for label in CLASS_INDICES if label in THRESHOLDS}  # CORRIGIDO: fora do while

    try:
        while True:
            raw_audio = record_audio()
            filtered_audio = preprocess_audio(raw_audio)
            # filtered_audio = remove_static_noise(filtered_audio, noise_profile)  # Desativado para testes

            if np.max(np.abs(filtered_audio)) < 0.01:
                print("Som muito fraco após filtro. Ignorado.")
                time.sleep(1)
                continue

            _, _, prediction = predict_sound(filtered_audio)
            history.add_prediction(prediction, filtered_audio)
            smoothed_pred = history.smooth()

            top_class_idx = np.argmax(smoothed_pred)
            top_class_name = get_class_name(top_class_idx)
            confidence = smoothed_pred[top_class_idx]
            top3_indices = smoothed_pred.argsort()[-3:][::-1]
            print("Top 3 classes:")
            for idx in top3_indices:
                 name = get_class_name(idx)
                 conf = smoothed_pred[idx]
            print(f"- {name}: {conf:.2%}")
            print(f"Som predominante: {top_class_name} ({confidence:.2%})")

            generate_spectrogram(filtered_audio, SAMPLE_RATE, smoothed_pred)  # Mostrar sempre para debug

            if recognize_keyword(filtered_audio, SAMPLE_RATE):
                print("🆘 Palavra-chave 'ajuda' detectada!")
                save_event_audio(filtered_audio, SAMPLE_RATE, "ajuda", 1)  # Salvando como evento especial
                # Você pode adicionar outras ações aqui, como enviar alerta, tocar som, etc.

            for label, idx in CLASS_INDICES.items():
                if label in THRESHOLDS and smoothed_pred[idx] > THRESHOLDS[label]:
                    event_counts[label] += 1
                    print(f"\n🔍 {label.upper()} DETECTADO (Confiança: {smoothed_pred[idx]:.2%}) [Total: {event_counts[label]}]")
                    save_event_audio(filtered_audio, SAMPLE_RATE, label.lower(), event_counts[label])

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nSessão encerrada pelo usuário.")
        for label, count in event_counts.items():
            print(f"{label}: {count} eventos detectados")

def get_class_name(class_index):
    for name, idx in CLASS_INDICES.items():
        if idx == class_index:
            return name
    return 'Other'

if __name__ == "__main__":
    main()