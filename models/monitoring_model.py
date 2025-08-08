from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from db.base import Base
import datetime


class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    noise_profile = Column(String, nullable=False)  # Ex: "silencioso", "moderado", "ruidoso"
    start_time = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    end_time = Column(DateTime, nullable=True)
    
    events = relationship("DetectedEvent", back_populates="session")
    user = relationship("User")


class DetectedEvent(Base):
    __tablename__ = "detected_events"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("monitoring_sessions.id"), nullable=False)
    event_type = Column(String, nullable=False)  # Ex: "Tosse", "Espirro"
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    session = relationship("MonitoringSession", back_populates="events")
