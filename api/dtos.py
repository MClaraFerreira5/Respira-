from pydantic import BaseModel
from datetime import datetime

class UserDto(BaseModel):
    name: str
    email: str
    password: str


class LoginDto(BaseModel):
    email: str
    password: str

class SessionStartResponse(BaseModel):
    session_id: int
    ambiente: str
    data_hora_inicio: datetime

class MonitoringResponse(BaseModel):
    status: str
    detalhes: str | None = None

class SessionReportResponse(BaseModel):
    session_id: int
    ambiente: str
    quantidade_tosse: int
    quantidade_espirro: int
    outros_eventos: int
    data_hora_inicio: str
    data_hora_fim: str
