from pydantic import BaseModel, Field, EmailStr, field_validator
from datetime import datetime
from typing import List
import re

class UserDto(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if not re.search(r'[A-Z]', v): raise ValueError('A senha precisa de uma letra maiúscula')
        if not re.search(r'[0-9]', v): raise ValueError('A senha precisa de um número')
        return v

class LoginDto(BaseModel):
    email: EmailStr
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

class SessionReportDetail(BaseModel):
    """Representa os dados agregados de uma única sessão em um relatório."""
    session_id: int
    ambiente_predominante: str = Field(..., alias="ambientePredominante")
    duracao_total_sessao_minutos: int = Field(..., alias="duracaoTotalSessaoMinutos")
    quantidade_tosse: int = Field(..., alias="quantidadeTosse")
    quantidade_espirro: int = Field(..., alias="quantidadeEspirro")
    outros_eventos: int = Field(..., alias="outrosEventos")
    data_hora_inicio: datetime = Field(..., alias="dataHoraInicio")
    data_hora_fim: datetime | None = Field(..., alias="dataHoraFim")

    class Config:
        populate_by_name = True
        orm_mode = True

class AggregatedReport(BaseModel):
    periodo_inicio: str = Field(..., alias="periodoInicio")
    periodo_fim: str = Field(..., alias="periodoFim")
    total_sessoes: int = Field(..., alias="totalSessoes")
    total_tosse: int = Field(..., alias="totalTosse")
    total_espirro: int = Field(..., alias="totalEspirro")
    total_outros_eventos: int = Field(..., alias="totalOutrosEventos")
    sessoes: List[SessionReportDetail]

    class Config:
        populate_by_name = True
