import numpy as np
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from collections import Counter

from models.monitoring_model import MonitoringSession, DetectedEvent
from services.prediction_recognition.prediction import predict_sound, get_class_name, CLASS_INDICES


def get_noise_profile_from_audio(audio_data: np.ndarray) -> str:
    rms_energy = np.sqrt(np.mean(audio_data**2))
    
    if rms_energy < 0.01:
        return "silencioso"
    elif rms_energy < 0.1:
        return "moderado"
    else:
        return "ruidoso"


def create_db_session(db: Session, user_id: int) -> MonitoringSession:
    """Cria uma nova sessão de monitoramento no banco de dados."""
    session = MonitoringSession(user_id=user_id, noise_profile="desconhecido")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def update_db_session_noise_profile(db: Session, session: MonitoringSession, noise_profile: str) -> MonitoringSession:
    session.noise_profile = noise_profile
    db.commit()
    db.refresh(session)
    return session

def get_active_db_session(db: Session, session_id: int, user_id: int) -> MonitoringSession | None:
    return db.query(MonitoringSession).filter(
        MonitoringSession.id == session_id,
        MonitoringSession.user_id == user_id,
        MonitoringSession.end_time == None
    ).first()

def close_db_session(db: Session, session: MonitoringSession) -> MonitoringSession:
    session.end_time = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    return session

def record_detected_events(db: Session, session_id: int, audio_data: np.ndarray):
    _, _, prediction = predict_sound(audio_data)
    
    EVENT_THRESHOLD = 0.2 
    event_map = {
        'Cough': 'Tosse',
        'Sneeze': 'Espirro'
    }

    detected_events_to_add = []
    for event_key, event_name in event_map.items():
        class_index = CLASS_INDICES.get(event_key)
        if class_index and prediction[class_index] > EVENT_THRESHOLD:
            event = DetectedEvent(session_id=session_id, event_type=event_name)
            detected_events_to_add.append(event)
    
    if detected_events_to_add:
        db.add_all(detected_events_to_add)
        db.commit()


def generate_session_report(session: MonitoringSession) -> dict:
    """
    Gera um relatório agregado a partir dos eventos já carregados na sessão.
    """
    event_counts = Counter(event.event_type for event in session.events)
    
    tosse_count = event_counts.get('Tosse', 0)
    espirro_count = event_counts.get('Espirro', 0)
    outros_count = sum(count for event_type, count in event_counts.items() 
                       if event_type not in ['Tosse', 'Espirro'])

    end_time = session.end_time if session.end_time else datetime.now(timezone.utc)

    return {
        "session_id": session.id,
        "ambiente": session.noise_profile,
        "quantidade_tosse": tosse_count,
        "quantidade_espirro": espirro_count,
        "outros_eventos": outros_count,
        "data_hora_inicio": session.start_time.isoformat(),
        "data_hora_fim": end_time.isoformat()
    }
