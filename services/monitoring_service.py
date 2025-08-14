from collections import Counter
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session, selectinload

from models.monitoring_model import MonitoringSession, DetectedEvent
from services.prediction_recognition.prediction import predict_sound, CLASS_INDICES


def get_noise_profile_from_audio(audio_data: np.ndarray) -> str:
    rms_energy = np.sqrt(np.mean(audio_data ** 2))

    if rms_energy < 0.01:
        return "silencioso"
    elif rms_energy < 0.02:
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


def get_sessions_by_date_range(db: Session, user_id: int, start_date: datetime, end_date: datetime):
    """Busca todas as sessões finalizadas de um usuário em um intervalo de datas."""
    return db.query(MonitoringSession).options(
        selectinload(MonitoringSession.events)
    ).filter(
        user_id == MonitoringSession.user_id,
        MonitoringSession.start_time >= start_date,
        MonitoringSession.start_time < end_date,
        MonitoringSession.end_time != None
    ).order_by(MonitoringSession.start_time.desc()).all()


def format_duration(start_time, end_time):
    if not start_time or not end_time:
        return 0
    duration_seconds = (end_time - start_time).total_seconds()
    return int(duration_seconds / 60)


def aggregate_sessions_data(sessions: list[MonitoringSession]) -> list[dict]:
    report_items = []
    for session in sessions:
        event_counts = Counter(event.event_type for event in session.events)
        item = {
            "session_id": session.id,
            "ambiente_predominante": session.noise_profile,
            "duracao_total_sessao_minutos": format_duration(session.start_time, session.end_time),
            "quantidade_tosse": event_counts.get('Tosse', 0),
            "quantidade_espirro": event_counts.get('Espirro', 0),
            "outros_eventos": sum(
                count for event_type, count in event_counts.items() if event_type not in ['Tosse', 'Espirro']),
            "data_hora_inicio": session.start_time,
            "data_hora_fim": session.end_time
        }
        report_items.append(item)
    return report_items
