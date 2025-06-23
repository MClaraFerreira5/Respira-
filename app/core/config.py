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
