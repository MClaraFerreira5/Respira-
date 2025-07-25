import numpy as np
import tensorflow_hub as hub
from app.core.config import MODEL_URL, CLASS_INDICES, THRESHOLDS, COUGH_PATTERN_COUNT

model = hub.load(MODEL_URL)


def predict_sound(audio):
    """Executa a predição usando YAMNet com tratamento de erros"""

    # Converte para o formato esperado pelo modelo
    waveform = audio / (np.max(np.abs(audio)) + 1e-8)
    waveform = waveform.astype(np.float32)

    # Executa a predição
    scores, embeddings, spectrogram = model(waveform)

    # Obtém as classes com maior pontuação (média das janelas temporais)
    prediction = np.mean(scores, axis=0)
    top_class = np.argmax(prediction)
    confidence = prediction[top_class]

    return top_class, confidence, prediction


def is_event(prediction, label):
    """"Refatoração de `is_sneeze`, `is_snore`, `is_gasp`"""
    idx = CLASS_INDICES[label]
    return prediction[idx] > THRESHOLDS[label]


def is_cough(prediction, history=None):
    """Verifica se o som detectado é tosse com checagem de contexto"""

    cough_score = prediction[CLASS_INDICES['Cough']]
    silence_score = prediction[CLASS_INDICES['Silence']]

    # Condições básicas
    above_threshold = cough_score > THRESHOLDS['Cough']
    not_silence = silence_score < 0.1

    # Verificação de padrão temporal se houver histórico
    is_pattern = False
    if history is not None and len(history.predictions) >= COUGH_PATTERN_COUNT:
        recent_coughs = sum(1 for p in history.predictions[-COUGH_PATTERN_COUNT:]
                            if p[CLASS_INDICES['Cough']] > THRESHOLDS['Cough'])
        is_pattern = recent_coughs >= COUGH_PATTERN_COUNT - 1

    return above_threshold and not_silence and (is_pattern or cough_score > 0.5)


def get_class_name(class_index):
    """Obtém o nome da classe a partir do índice"""
    for name, idx in CLASS_INDICES.items():
        if idx == class_index:
            return name
    return 'Other'
