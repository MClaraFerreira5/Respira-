import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    POSTGRES_DB: str = "respira_mais"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"

    POSTGRES_USER: str = os.environ.get("DB_USER", "")
    POSTGRES_PASSWORD: str = os.environ.get("DB_PASSWORD", "")

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    @property
    def DATABASE_URL(self):
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    class Config:
        env_file = ".env"


settings = Settings()

MODEL_URL = 'https://tfhub.dev/google/yamnet/1'
SAMPLE_RATE = 16000  # YAMNet requer áudio em 16kHz
RECORD_DURATION = 3  # Duração da gravação em segundos
THRESHOLDS = {  # Modificação: thresholds específicos por classe
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
