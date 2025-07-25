import streamlit as st
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from datetime import datetime
from scipy import signal
import speech_recognition as sr
from collections import deque
import time
from datetime import datetime

from app.services.audio_history import AudioHistory
from app.services.audio_processing import preprocess_audio
from app.services.prediction import predict_sound, is_cough, get_class_name, is_event
from app.services.visualization import generate_spectrogram
from app.services.utils import save_event_audio
from app.services.noise_processing import capture_noise_profile
from app.services.speech_recognition import recognize_keyword
from app.core.config import SAMPLE_RATE, RECORD_DURATION, CLASS_INDICES, THRESHOLDS

SAMPLE_RATE = 16000
RECORD_DURATION = 3 

MODEL_URL = 'https://tfhub.dev/google/yamnet/1'

CLASS_INDICES = {
    'Cough': 42, 'Snore': 38, 'Sneeze': 44, 'Sniff': 45, 'Gasp': 39
}

THRESHOLD_PROFILES = {
    'low_noise': {'Cough': 0.40, 'Snore': 0.20, 'Sneeze': 0.45, 'Gasp': 0.20, 'Sniff': 0.20},
    'medium_noise': {'Cough': 0.55, 'Snore': 0.30, 'Sneeze': 0.60, 'Gasp': 0.30, 'Sniff': 0.30},
    'high_noise': {'Cough': 0.70, 'Snore': 0.45, 'Sneeze': 0.75, 'Gasp': 0.45, 'Sniff': 0.45}
}

PATTERN_WINDOW_SIZE = 5
KEYWORDS = ["ajuda", "socorro"]
CRITICAL_EVENTS = ['Gasp']
PRIORITIZE_OVER_COUGH_SNEEZE = ['Snore', 'Sniff', 'Sneeze']

@st.cache_resource
def load_yamnet_model():
    try:
        model = hub.load(MODEL_URL)
        st.success("✅ Modelo YAMNet carregado com sucesso!")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo YAMNet: {e}")
        st.error("Verifique sua conexão com a internet e as configurações do TensorFlow Hub.")
        return None

model = load_yamnet_model()

class AudioHistory:
    
    def __init__(self, max_length=15):
        self.predictions = deque(maxlen=max_length)
        self.noise_levels = deque(maxlen=10)

    def add_prediction(self, prediction, noise_level=None):
        self.predictions.append(prediction)
        if noise_level is not None:
            self.noise_levels.append(noise_level)

    def smooth(self):
        if not self.predictions:
            return np.zeros(521)
        return np.mean(self.predictions, axis=0)

    def get_current_noise_profile(self):
        if not self.noise_levels:
            return 'medium_noise'
        avg_noise = np.mean(self.noise_levels)
        if avg_noise < 0.03: return 'low_noise'
        if avg_noise < 0.09: return 'medium_noise'
        return 'high_noise'

class RespiratoryEventDetector:
    def __init__(self):
        self.history = AudioHistory()
        self.event_counts = {label: 0 for label in CLASS_INDICES}
        self.current_thresholds = THRESHOLD_PROFILES['medium_noise']
        self.noise_profile_audio = None
        self.session_events_log = []

    def update_noise_profile(self, noise_audio):
        self.noise_profile_audio = noise_audio
        noise_level = self.calculate_audio_energy(noise_audio)
        self.history.noise_levels.append(noise_level)
        noise_class = self.history.get_current_noise_profile()
        self.current_thresholds = THRESHOLD_PROFILES[noise_class]
        return noise_class, noise_level

    def calculate_audio_energy(self, audio):
        return np.sqrt(np.mean(audio**2))

    def remove_static_noise(self, audio):
        if self.noise_profile_audio is None or len(self.noise_profile_audio) == 0:
            return audio
        noise_energy = self.calculate_audio_energy(self.noise_profile_audio)
        audio_energy = self.calculate_audio_energy(audio)
        if audio_energy < 1.8 * noise_energy: # Threshold mais agressivo
            return np.zeros_like(audio)
        return audio

    def preprocess_audio(self, audio):
        if np.max(np.abs(audio)) < 0.001: return np.zeros_like(audio)
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Filtro passa-banda para focar em frequências da voz e respiração
        lowcut, highcut = 80, 6000
        nyquist = 0.5 * SAMPLE_RATE
        b, a = signal.butter(5, [lowcut/nyquist, highcut/nyquist], btype='band')
        audio = signal.filtfilt(b, a, audio)
        
        return audio.astype(np.float32)
        
    def detect_most_probable_event(self, prediction):
        potential_events = []
        for event_class, threshold in self.current_thresholds.items():
            class_score = prediction[CLASS_INDICES[event_class]]
            if class_score > threshold:
                potential_events.append((event_class, class_score))

        if not potential_events:
            return None, 0.0

        # Lógica de Priorização
        critical_detected = [e for e in potential_events if e[0] in CRITICAL_EVENTS]
        if critical_detected:
            return max(critical_detected, key=lambda x: x[1])

        other_relevant = [e for e in potential_events if e[0] in PRIORITIZE_OVER_COUGH_SNEEZE]
        cough_sneeze = [e for e in potential_events if e[0] in ['Cough', 'Sneeze']]

        if other_relevant and cough_sneeze:
            best_other = max(other_relevant, key=lambda x:x[1])
            best_cough = max(cough_sneeze, key=lambda x:x[1])
            # Prioriza outros eventos se a confiança for comparável à da tosse/espirro
            return best_other if best_other[1] >= best_cough[1] * 0.8 else best_cough
        
        return max(potential_events, key=lambda x: x[1])

def record_audio(duration, message_placeholder):
    try:
        message_placeholder.info(f"🎙️ Gravando por {duration} segundos...")
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')

def record_audio(duration=RECORD_DURATION, sample_rate=SAMPLE_RATE):
    try:
        print(f"Gravando {duration} segundos de áudio...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        message_placeholder.empty()
        return np.squeeze(audio)
    except Exception as e:
        st.error(f"Erro na gravação de áudio: {e}. Verifique o microfone.")
        return np.zeros(int(duration * SAMPLE_RATE))

def predict_sound(audio, model_yamnet):
    try:
        waveform = tf.constant(audio, dtype=tf.float32)
        scores, _, _ = model_yamnet(waveform)
        return np.mean(scores, axis=0)
    except Exception as e:
        st.warning(f"Erro na predição: {e}")
        return np.zeros(521)

def generate_spectrogram_plot(audio, prediction, event):
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1, 2]})

    axes[0].specgram(audio, Fs=SAMPLE_RATE, cmap=cm.viridis, NFFT=512, noverlap=256)
    axes[0].set_title(f'Espectrograma - Evento Detectado: {event}')
    axes[0].set_ylabel('Frequência (Hz)')

    time_axis = np.linspace(0, len(audio) / SAMPLE_RATE, num=len(audio))
    axes[1].plot(time_axis, audio)
    axes[1].set_title('Forma de Onda')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)

    classes = list(CLASS_INDICES.keys())
    values = [prediction[CLASS_INDICES[c]] for c in classes]
    colors = ['#ff6347' if c in CRITICAL_EVENTS else '#1f77b4' for c in classes] # Vermelho para eventos críticos

    bars = axes[2].bar(classes, values, color=colors)

    axes[2].set_title('Probabilidades das Classes de Interesse')
    axes[2].set_ylabel('Confiança')
    axes[2].set_ylim(0, 1)

    plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right")

    if event in CLASS_INDICES:
        threshold_val = st.session_state.detector.current_thresholds.get(event, 0)
        axes[2].axhline(y=threshold_val, color='r', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold_val:.2f})')
        axes[2].legend()

    for bar in bars:
        height = bar.get_height()
        axes[2].annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # Deslocamento vertical de 3 pontos
                         textcoords="offset points",
                         ha='center', va='bottom')

    plt.tight_layout()
    return fig
        print(f"Erro ao gravar áudio: {e}")
        return np.zeros(int(duration * sample_rate))


def main():
    print("\n=== Sistema Avançado de Detecção de Áudio ===")
    history = AudioHistory(max_length=10)
    session_start = datetime.now()
    noise_profile = capture_noise_profile()
    event_counts = {label: 0 for label in CLASS_INDICES if label in THRESHOLDS}


@st.cache_data
def recognize_keyword_cached(audio_segment, sample_rate):
    r = sr.Recognizer()
    audio_int16 = (audio_segment * 32767).astype(np.int16)
    audio_data = sr.AudioData(audio_int16.tobytes(), sample_rate, 2)
    try:

        text = r.recognize_google(audio_data, language='pt-BR')
        for keyword in KEYWORDS:
            if keyword.lower() in text.lower():
                return text, True # Retorna texto e flag de detecção
        return text, False
    except sr.UnknownValueError:
        return "Não foi possível entender o áudio.", False
    except sr.RequestError as e:
        return f"Erro de serviço de fala: {e}", False

st.set_page_config(page_title="Monitoramento Respiratório", layout="wide")
st.title("🫁 Monitoramento Respiratório Avançado")
st.markdown("Uma aplicação para detectar e analisar sons respiratórios usando YAMNet com lógica robusta de detecção.")

# Inicializa o detector na sessão do Streamlit
if 'detector' not in st.session_state:
    st.session_state.detector = RespiratoryEventDetector()

detector = st.session_state.detector

st.sidebar.header("Controles")
col1, col2 = st.sidebar.columns(2)

status_placeholder = st.sidebar.empty()
analysis_placeholder = st.empty()
record_msg_placeholder = st.sidebar.empty()

with col1:
    if st.button("📢 Capturar Ruído", help="Grave o som ambiente para calibrar o detector."):
        with st.spinner('Gravando ruído ambiente...'):
            noise_audio = record_audio(RECORD_DURATION, record_msg_placeholder)
            noise_class, noise_level = detector.update_noise_profile(noise_audio)
            status_placeholder.success(f"Perfil de ruído: **{noise_class.upper()}** (Nível: {noise_level:.4f})")

with col2:
    if st.button("🔎 Analisar Áudio", type="primary", help="Grave e analise um novo segmento de áudio."):
        if model is None:
            st.error("Modelo não carregado. Não é possível analisar.")
        elif detector.noise_profile_audio is None:
            status_placeholder.warning("⚠️ Capture o perfil de ruído primeiro.")
        else:
            with analysis_placeholder.container():
                st.info("Iniciando análise...")
                
                # 1. Gravação e Pré-processamento
                raw_audio = record_audio(RECORD_DURATION, record_msg_placeholder)
                processed_audio = detector.preprocess_audio(raw_audio)
                final_audio = detector.remove_static_noise(processed_audio)
                
                # Verifica se o áudio não é apenas silêncio após o processamento
                if np.max(np.abs(final_audio)) < 0.01:
                    st.info("🔇 Nenhum evento significativo detectado (sinal muito fraco após filtragem).")
                else:
                    # 2. Predição e Detecção
                    prediction = predict_sound(final_audio, model)
                    noise_level = detector.calculate_audio_energy(final_audio)
                    detector.history.add_prediction(prediction, noise_level)
                    smoothed_pred = detector.history.smooth()
                    
                    event, confidence = detector.detect_most_probable_event(smoothed_pred)

                    # 3. Exibição dos Resultados
                    st.subheader("Resultados da Análise")
                    res_col1, res_col2 = st.columns([1, 2])

                    with res_col1:
                        if event:
                            st.success(f"**Evento Principal:** `{event}`")
                            st.metric(label="Confiança", value=f"{confidence:.2%}")
                            detector.event_counts[event] += 1
                            
                            # Salva e exibe áudio
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            os.makedirs("events", exist_ok=True)
                            filename = f"events/{event}_{timestamp}.wav"
                            sf.write(filename, final_audio, SAMPLE_RATE)
                            st.audio(filename)

                        else:
                            st.info("Nenhum evento respiratório específico detectado acima do threshold.")
                        
                        # Análise de Palavra-chave
                        st.markdown("---")
                        st.write("**Reconhecimento de Fala**")
                        with st.spinner("Analisando palavras-chave..."):
                            transcription, found = recognize_keyword_cached(final_audio, SAMPLE_RATE)
                            if found:
                                st.error(f"🚨 PALAVRA-CHAVE DETECTADA: '{transcription}'")
                            else:
                                st.caption(f"Transcrição: *{transcription}*")
                                
                    with res_col2:
                        fig = generate_spectrogram_plot(final_audio, smoothed_pred, event or "Nenhum")
                        st.pyplot(fig)
                    
                    # Adiciona ao log da sessão
                    log_entry = {
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'event': event or "Nenhum",
                        'confidence': f"{confidence:.1%}" if event else "N/A"
                    }
                    detector.session_events_log.insert(0, log_entry)


st.sidebar.markdown("---")
st.sidebar.header("Status e Histórico")

if detector.noise_profile_audio is not None:
    noise_class = detector.history.get_current_noise_profile()
    status_placeholder.success(f"Perfil de ruído: **{noise_class.upper()}**")
else:
    status_placeholder.info("Aguardando captura do perfil de ruído.")

st.sidebar.subheader("Contagem de Eventos")
st.sidebar.json(detector.event_counts)

st.sidebar.subheader("Log da Sessão")
st.sidebar.dataframe(detector.session_events_log, use_container_width=True)
        while True:
            audio = record_audio()
            processed = preprocess_audio(audio)

            if np.max(np.abs(processed)) < 0.01:
                print("Som muito fraco após pré-processamento. Ignorado.")
                time.sleep(1)
                continue

            _, _, prediction = predict_sound(processed)
            history.add_prediction(prediction, processed)
            smoothed_pred = history.smooth()

            top_class_idx = np.argmax(smoothed_pred)
            top_class_name = get_class_name(top_class_idx)
            confidence = smoothed_pred[top_class_idx]

            print(f"Som predominante: {top_class_name} ({confidence:.2%})")

            generate_spectrogram(processed, smoothed_pred)

            if recognize_keyword(processed, SAMPLE_RATE):
                print("🆘 Palavra-chave 'ajuda' detectada!")
                save_event_audio(processed, SAMPLE_RATE, "ajuda", 1)

            for label in THRESHOLDS:
                if is_event(smoothed_pred, label):
                    event_counts[label] += 1
                    print(
                        f"\n🔍 {label.upper()} DETECTADO (Confiança: {smoothed_pred[CLASS_INDICES[label]]:.2%}) Total: {event_counts[label]}")
                    save_event_audio(processed, SAMPLE_RATE, label.lower(), event_counts[label])

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nSessão encerrada pelo usuário.")
        for label, count in event_counts.items():
            print(f"{label}: {count} eventos detectados")


if __name__ == "__main__":
    main()

# def main():
#     print("\n=== Sistema Avançado de Detecção de Tosse ===")
#     print("Pressione Ctrl+C para sair\n")
#
#     history = AudioHistory(max_length=10)
#     #cough_count = 0
#     #sneeze_count = 0
#     #snore_count = 0
#     #gasp_count = 0
#     session_start = datetime.now()
#
#     noise_profile = capture_noise_profile()
#     event_counts = {label: 0 for label in CLASS_INDICES if label in THRESHOLDS}  # CORRIGIDO: fora do while
#
#     try:
#         while True:
#             raw_audio = record_audio()
#             filtered_audio = preprocess_audio(raw_audio)
#             # filtered_audio = remove_static_noise(filtered_audio, noise_profile)  # Desativado para testes
#
#             if np.max(np.abs(filtered_audio)) < 0.01:
#                 print("Som muito fraco após filtro. Ignorado.")
#                 time.sleep(1)
#                 continue
#
#             _, _, prediction = predict_sound(filtered_audio)
#             history.add_prediction(prediction, filtered_audio)
#             smoothed_pred = history.smooth()
#
#             top_class_idx = np.argmax(smoothed_pred)
#             top_class_name = get_class_name(top_class_idx)
#             confidence = smoothed_pred[top_class_idx]
#             top3_indices = smoothed_pred.argsort()[-3:][::-1]
#             print("Top 3 classes:")
#             for idx in top3_indices:
#                 name = get_class_name(idx)
#                 conf = smoothed_pred[idx]
#             print(f"- {name}: {conf:.2%}")
#             print(f"Som predominante: {top_class_name} ({confidence:.2%})")
#
#             generate_spectrogram(filtered_audio, SAMPLE_RATE, smoothed_pred)  # Mostrar sempre para debug
#
#             if recognize_keyword(filtered_audio, SAMPLE_RATE):
#                 print("🆘 Palavra-chave 'ajuda' detectada!")
#                 save_event_audio(filtered_audio, SAMPLE_RATE, "ajuda", 1)  # Salvando como evento especial
#                 # Você pode adicionar outras ações aqui, como enviar alerta, tocar som, etc.
#
# for label, idx in CLASS_INDICES.items(): if label in THRESHOLDS and smoothed_pred[idx] > THRESHOLDS[label]:
# event_counts[label] += 1 print( f"\n🔍 {label.upper()} DETECTADO (Confiança: {smoothed_pred[idx]:.2%}) [Total: {
# event_counts[label]}]") save_event_audio(filtered_audio, SAMPLE_RATE, label.lower(), event_counts[label])
#
#             time.sleep(0.5)
#
#     except KeyboardInterrupt:
#         print("\nSessão encerrada pelo usuário.")
#         for label, count in event_counts.items():
#             print(f"{label}: {count} eventos detectados")

# # Configurações
# MODEL_URL = 'https://tfhub.dev/google/yamnet/1'
# SAMPLE_RATE = 16000  # YAMNet requer áudio em 16kHz
# RECORD_DURATION = 3  # Duração da gravação em segundos
# THRESHOLDS = {        # Modificação: thresholds específicos por classe
#     'Cough': 0.15,
#     'Snore': 0.15,
#     'Breathing': 0.15,
#     'Sneeze': 0.15,
#     'Gasp': 0.15,
#     'Speech': 0.15
# }
#
# MIN_COUGH_DURATION = 0.2  # Duração mínima de uma tosse em segundos
# COUGH_PATTERN_COUNT = 3  # Número mínimo de tosses para caracterizar um padrão
# KEYWORD = "ajuda"
#
# # Carrega o modelo YAMNet
# model = hub.load(MODEL_URL)
#
# # Índices das classes no YAMNet
# CLASS_INDICES = {
#     'Cough': 42,
#     'Snore': 38,
#     'Breathing': 36,
#     'Speech': 65,
#     'Silence': 1,
#     'Sneeze': 44,
#     'Sniff': 45,
#     'Gasp': 39
# }

# class AudioHistory:
#     def __init__(self, max_length=10):
#         self.max_length = max_length
#         self.predictions = []
#         self.audio_segments = []
#         self.timestamps = []
#
#     def add_prediction(self, prediction, audio_segment=None):
#         self.predictions.append(prediction)
#         self.audio_segments.append(audio_segment)
#         self.timestamps.append(datetime.now())
#
#         if len(self.predictions) > self.max_length:
#             self.predictions.pop(0)
#             self.audio_segments.pop(0)
#             self.timestamps.pop(0)
#
#     def smooth(self):     # Modificação: suavização de predições
#         if not self.predictions:
#             return np.zeros(521)
#         return np.mean(self.predictions, axis=0)
#
#     def get_recent_pattern(self, window_size=3):
#         if len(self.predictions) >= window_size:
#             return self.predictions[-window_size:]
#         return None
#
#     def get_cough_count(self, threshold=0.3):
#         return sum(1 for p in self.predictions
#                    if p[CLASS_INDICES['Cough']] > threshold)

# def capture_noise_profile(duration=2):    # Modificação: captura de ruído ambiente
#     print("Capturando perfil de ruído de fundo...")
#     audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
#     sd.wait()
#     return np.squeeze(audio)
#
# def remove_static_noise(audio, noise_profile):
#     energy = lambda x: np.sqrt(np.mean(x**2))
#     if energy(noise_profile) == 0:
#         return audio
#     noise_energy = energy(noise_profile)
#     audio_energy = energy(audio)
#     if audio_energy < 1.5 * noise_energy:
#         return np.zeros_like(audio)
#     return audio

# def preprocess_audio(audio):
#     audio = audio / (np.max(np.abs(audio)) + 1e-8)
#     lowcut = 50
#     highcut = 3000
#     nyquist = 0.5 * SAMPLE_RATE
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = signal.butter(5, [low, high], btype='band')
#     audio = signal.filtfilt(b, a, audio)
#
#     energy = np.sqrt(np.mean(audio ** 2))
#     threshold = energy * 0.05
#     valid_samples = np.where(np.abs(audio) > threshold)[0]
#
#     if len(valid_samples) > 0:
#         start = max(0, valid_samples[0] - 100)
#         end = min(len(audio), valid_samples[-1] + 100)
#         audio = audio[start:end]
#
#     return audio.astype(np.float32)


# def generate_spectrogram(audio, sample_rate=SAMPLE_RATE, prediction=None):
#     plt.figure(figsize=(12, 8))
#
#     plt.subplot(3, 1, 1)
#     spec, freqs, times, im = plt.specgram(audio, Fs=sample_rate,
#                                           cmap=cm.viridis, NFFT=512,
#                                           noverlap=256)
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Espectrograma do Áudio')
#     plt.xlabel('Tempo (s)')
#     plt.ylabel('Frequência (Hz)')
#
#     plt.subplot(3, 1, 2)
#     time_axis = np.linspace(0, len(audio) / sample_rate, num=len(audio))
#     plt.plot(time_axis, audio)
#     plt.title('Forma de Onda')
#     plt.xlabel('Tempo (s)')
#     plt.ylabel('Amplitude')
#     plt.grid(True)
#
#     if prediction is not None:
#         plt.subplot(3, 1, 3)
#         classes = list(CLASS_INDICES.keys())
#         values = [prediction[idx] for idx in CLASS_INDICES.values()]
#         colors = ['red' if cls == 'Cough' else 'blue' for cls in classes]
#         bars = plt.bar(classes, values, color=colors)
#         plt.title('Probabilidades das Classes')
#         plt.ylabel('Probabilidade')
#         plt.xticks(rotation=45)
#         plt.ylim(0, 1)
#
#         for bar in bars:
#             height = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width() / 2., height,
#                      f'{height:.2f}',
#                      ha='center', va='bottom')
#
#     plt.tight_layout()
#     plt.show()

# def predict_sound(audio):
#     try:
#         waveform = audio / (np.max(np.abs(audio)) + 1e-8)
#         waveform = waveform.astype(np.float32)
#         scores, embeddings, spectrogram = model(waveform)
#         prediction = np.mean(scores, axis=0)
#         top_class = np.argmax(prediction)
#         confidence = prediction[top_class]
#         return top_class, confidence, prediction
#     except Exception as e:
#         print(f"Erro na predição: {e}")
#         return -1, 0, np.zeros(521)
#
# def is_cough(prediction, history=None):
#     cough_score = prediction[CLASS_INDICES['Cough']]
#     silence_score = prediction[CLASS_INDICES['Silence']]
#     above_threshold = cough_score > THRESHOLDS['Cough']
#     not_silence = silence_score < 0.1
#     is_pattern = False
#     if history is not None and len(history.predictions) >= COUGH_PATTERN_COUNT:
#         recent_coughs = sum(1 for p in history.predictions[-COUGH_PATTERN_COUNT:]
#                             if p[CLASS_INDICES['Cough']] > THRESHOLDS['Cough'])  # Fixed: Use specific threshold
#         is_pattern = recent_coughs >= COUGH_PATTERN_COUNT - 1
#     return above_threshold and not_silence and (is_pattern or cough_score > 0.5)

# def is_sneeze(prediction):
#     return prediction[CLASS_INDICES['Sneeze']] > THRESHOLDS['Sneeze']  # Fixed: Use specific threshold
#
# def is_snore(prediction):
#     return prediction[CLASS_INDICES['Snore']] > THRESHOLDS['Snore']  # Fixed: Use specific threshold
#
# def is_gasp(prediction):
#     return prediction[CLASS_INDICES['Gasp']] > THRESHOLDS['Gasp']  # Fixed: Use specific threshold

# def recognize_keyword(audio, sample_rate):
#     """Realiza reconhecimento de fala e verifica se palavra-chave está presente"""
#     try:
#         filename = "temp_speech.wav"
#         sf.write(filename, audio, sample_rate)
#         recognizer = sr.Recognizer()
#         with sr.AudioFile(filename) as source:
#             audio_data = recognizer.record(source)
#         transcript = recognizer.recognize_google(audio_data, language="pt-BR")
#         print(f"Transcrição: {transcript}")
#         return KEYWORD.lower() in transcript.lower()
#     except Exception as e:
#         print(f"Erro no reconhecimento de fala: {e}")
#         return False

# def save_event_audio(audio, sample_rate, label, count):
#     os.makedirs(f"{label}_detections", exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{label}_detections/{label}_{timestamp}_{count}.wav"
#     sf.write(filename, audio, sample_rate)
#     print(f"Áudio de {label} salvo como {filename}")


# def get_class_name(class_index):
#     for name, idx in CLASS_INDICES.items():
#         if idx == class_index:
#             return name
#     return 'Other'

