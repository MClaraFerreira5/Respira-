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

# Configura o FFmpeg para o PyDub
os.environ["PATH"] += os.pathsep + 'C:\\ffmpeg\\bin'

# Configurações
MODEL_URL = 'https://tfhub.dev/google/yamnet/1'
SAMPLE_RATE = 16000  # YAMNet requer áudio em 16kHz
RECORD_DURATION = 3  # Duração da gravação em segundos
THRESHOLD = 0.3  # Limiar de confiança para detecção inicial
MIN_COUGH_DURATION = 0.2  # Duração mínima de uma tosse em segundos
COUGH_PATTERN_COUNT = 3  # Número mínimo de tosses para caracterizar um padrão

# Carrega o modelo YAMNet
model = hub.load(MODEL_URL)

# Índices das classes no YAMNet
CLASS_INDICES = {
    'Cough': 42,
    'Snore': 38,
    'Breathing': 36,
    'Laughter': 13,
    'Speech': 0,
    'Silence': 1
}


class AudioHistory:
    def __init__(self, max_length=10):
        self.max_length = max_length
        self.predictions = []
        self.audio_segments = []
        self.timestamps = []

    def add_prediction(self, prediction, audio_segment):
        self.predictions.append(prediction)
        self.audio_segments.append(audio_segment)
        self.timestamps.append(datetime.now())

        if len(self.predictions) > self.max_length:
            self.predictions.pop(0)
            self.audio_segments.pop(0)
            self.timestamps.pop(0)

    def get_recent_pattern(self, window_size=3):
        if len(self.predictions) >= window_size:
            return self.predictions[-window_size:]
        return None

    def get_cough_count(self, threshold=THRESHOLD):
        return sum(1 for p in self.predictions
                   if p[CLASS_INDICES['Cough']] > threshold)


def preprocess_audio(audio):
    """Melhoria no pré-processamento do áudio"""
    # Normalização
    audio = audio / (np.max(np.abs(audio)) + 1e-8)  # Evita divisão por zero

    # Filtro passa-banda para focar em frequências relevantes para tosse (50Hz - 3000Hz)
    lowcut = 50
    highcut = 3000
    nyquist = 0.5 * SAMPLE_RATE
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    audio = signal.filtfilt(b, a, audio)

    # Remove silêncio inicial/final
    energy = np.sqrt(np.mean(audio ** 2))
    threshold = energy * 0.05  # Ajuste conforme necessário
    valid_samples = np.where(np.abs(audio) > threshold)[0]

    if len(valid_samples) > 0:
        start = max(0, valid_samples[0] - 100)
        end = min(len(audio), valid_samples[-1] + 100)
        audio = audio[start:end]

    return audio.astype(np.float32)


def record_audio(duration=RECORD_DURATION, sample_rate=SAMPLE_RATE):
    """Grava áudio do microfone com tratamento de erros"""
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
    """Gera visualização completa do áudio e predições"""
    plt.figure(figsize=(12, 8))

    # Espectrograma
    plt.subplot(3, 1, 1)
    spec, freqs, times, im = plt.specgram(audio, Fs=sample_rate,
                                          cmap=cm.viridis, NFFT=512,
                                          noverlap=256)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma do Áudio')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência (Hz)')

    # Forma de onda
    plt.subplot(3, 1, 2)
    time_axis = np.linspace(0, len(audio) / sample_rate, num=len(audio))
    plt.plot(time_axis, audio)
    plt.title('Forma de Onda')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Gráfico de probabilidades (se houver predição)
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

        # Adiciona valores nas barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def predict_sound(audio):
    """Executa a predição usando YAMNet com tratamento de erros"""
    try:
        # Converte para o formato esperado pelo modelo
        waveform = audio / (np.max(np.abs(audio)) + 1e-8)  # Normaliza
        waveform = waveform.astype(np.float32)

        # Executa a predição
        scores, embeddings, spectrogram = model(waveform)

        # Obtém as classes com maior pontuação (média das janelas temporais)
        prediction = np.mean(scores, axis=0)
        top_class = np.argmax(prediction)
        confidence = prediction[top_class]

        return top_class, confidence, prediction
    except Exception as e:
        print(f"Erro na predição: {e}")
        return -1, 0, np.zeros(len(CLASS_INDICES))


def is_cough(prediction, history=None):
    """Verifica se o som detectado é tosse com checagem de contexto"""
    cough_score = prediction[CLASS_INDICES['Cough']]
    silence_score = prediction[CLASS_INDICES['Silence']]

    # Condições básicas
    above_threshold = cough_score > THRESHOLD
    not_silence = silence_score < 0.1

    # Verificação de padrão temporal se houver histórico
    is_pattern = False
    if history is not None and len(history.predictions) >= COUGH_PATTERN_COUNT:
        recent_coughs = sum(1 for p in history.predictions[-COUGH_PATTERN_COUNT:]
                            if p[CLASS_INDICES['Cough']] > THRESHOLD)
        is_pattern = recent_coughs >= COUGH_PATTERN_COUNT - 1

    return above_threshold and not_silence and (is_pattern or cough_score > 0.5)


def save_cough_audio(audio, sample_rate, cough_count):
    """Salva o áudio da tosse detectada"""
    os.makedirs("cough_detections", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cough_detections/cough_{timestamp}_{cough_count}.wav"
    sf.write(filename, audio, sample_rate)
    print(f"Áudio da tosse salvo como {filename}")


def main():
    print("\n=== Sistema Avançado de Detecção de Tosse ===")
    print("Pressione Ctrl+C para sair\n")

    history = AudioHistory(max_length=10)
    cough_count = 0
    session_start = datetime.now()

    try:
        while True:
            # 1. Grava o áudio
            audio = record_audio()

            if np.max(np.abs(audio)) < 0.01:  # Verifica se há áudio válido
                print("Nenhum áudio detectado. Verifique o microfone.")
                time.sleep(1)
                continue

            # 2. Pré-processamento
            processed_audio = preprocess_audio(audio)

            # 3. Faz a predição
            top_class, confidence, prediction = predict_sound(processed_audio)
            history.add_prediction(prediction, processed_audio)

            # 4. Verifica se é tosse com contexto
            if is_cough(prediction, history):
                cough_prob = prediction[CLASS_INDICES['Cough']]
                cough_count += 1
                print(f"\n🚨 TOSSE DETECTADA! (Confiança: {cough_prob:.2%}) [Total: {cough_count}]")

                # Salva o áudio da tosse
                save_cough_audio(processed_audio, SAMPLE_RATE, cough_count)

                # Gera alerta sonoro (opcional)
                sd.play(0.2 * np.sin(2 * np.pi * 880 * np.linspace(0, 1, 2000)), samplerate=44100)
            else:
                class_name = get_class_name(top_class)
                print(f"\nSom detectado: {class_name} (Confiança: {confidence:.2%})")

            # 5. Gera visualização
            generate_spectrogram(processed_audio, SAMPLE_RATE, prediction)

            # Pequena pausa entre gravações
            time.sleep(0.5)

    except KeyboardInterrupt:
        session_duration = datetime.now() - session_start
        print(f"\nSessão encerrada após {session_duration}")
        print(f"Total de tosses detectadas: {cough_count}")
        print(f"Média de tosses por hora: {cough_count / (session_duration.total_seconds() / 3600):.1f}")


def get_class_name(class_index):
    """Obtém o nome da classe a partir do índice"""
    for name, idx in CLASS_INDICES.items():
        if idx == class_index:
            return name
    return 'Other'


if __name__ == "__main__":
    main()