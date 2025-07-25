import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from datetime import datetime
from scipy import signal
import speech_recognition as sr
from collections import deque

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