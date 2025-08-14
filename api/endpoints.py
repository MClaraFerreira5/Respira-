import io
from datetime import datetime, timedelta, timezone

import librosa
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.dtos import (UserDto, LoginDto, SessionStartResponse, MonitoringResponse, SessionReportResponse,
                      AggregatedReport)
from core.dependencies import get_current_user
from db.session import get_db
from services import monitoring_service
from services.auth.auth import create_access_token
from services.log.logger import log_event
from services.processing.audio_processing import preprocess_audio
from services.user import user_service

router = APIRouter()

auth_router = APIRouter(prefix="/auth", tags=["Autenticação"])


@auth_router.post("/register")
def register(userResponse: UserDto, db: Session = Depends(get_db)):
    # log_event(f"Dados recebidos na API: {userResponse}")
    existing = user_service.login_user(db, userResponse.email, userResponse.password)
    if existing:
        raise HTTPException(status_code=400, detail="Esse email já foi registrado")
    user = user_service.register_user(db, userResponse.name, userResponse.email, userResponse.password)
    access_token = create_access_token(data={"sub": str(user.id)})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }


@auth_router.post("/login")
def login(loginDto: LoginDto, db: Session = Depends(get_db)):
    user = user_service.login_user(db, loginDto.email, loginDto.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    access_token = create_access_token(data={"sub": str(user.id)})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }


monitoring_router = APIRouter(prefix="/monitoramento", tags=["Monitoramento"])


@monitoring_router.post("/iniciar_sessao", response_model=SessionStartResponse)
def iniciar_sessao_monitoramento(
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user)
):
    """
    Inicia uma nova sessão de monitoramento para o usuário logado.
    Esta rota deve ser chamada uma vez no início do monitoramento.
    """
    session = monitoring_service.create_db_session(db, user_id=user_id)
    log_event(f"Sessão {session.id} iniciada para o usuário {user_id}.")

    return SessionStartResponse(
        session_id=session.id,
        ambiente=session.noise_profile,
        data_hora_inicio=session.start_time
    )


@monitoring_router.post("/analisar_ambiente", response_model=MonitoringResponse)
async def analisar_ambiente(
        session_id: int = Query(...),
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user)
):
    """
    Recebe um áudio para analisar o perfil de ruído do ambiente
    e atualiza a sessão de monitoramento ativa.
    """
    active_session = monitoring_service.get_active_db_session(db, session_id=session_id, user_id=user_id)
    if not active_session:
        raise HTTPException(status_code=404, detail="Sessão de monitoramento não encontrada, expirada ou inválida.")

    try:
        contents = await file.read()
        audio, sr = librosa.load(io.BytesIO(contents), sr=16000, mono=True)

        noise_profile = monitoring_service.get_noise_profile_from_audio(audio)
        monitoring_service.update_db_session_noise_profile(db, active_session, noise_profile)

        log_event(f"Ambiente da sessão {session_id} atualizado para '{noise_profile}'")

        return MonitoringResponse(status="ok", detalhes=f"Perfil de ruído atualizado para {noise_profile}.")
    except Exception as e:
        log_event(f"Erro ao analisar ambiente: {e}", to_file=True)
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")


@monitoring_router.post("/audio", response_model=MonitoringResponse)
async def monitorar_audio(
        session_id: int = Query(...),
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user)
):
    active_session = monitoring_service.get_active_db_session(db, session_id=session_id, user_id=user_id)
    if not active_session:
        raise HTTPException(status_code=404, detail="Sessão de monitoramento não encontrada, expirada ou inválida.")

    try:
        contents = await file.read()
        audio, sr = librosa.load(io.BytesIO(contents), sr=16000, mono=True)

        log_event(f"Sample rate: {sr} Hz")
        log_event(f"Formato interno: {audio.dtype}")
        log_event(f"Número de canais: {audio.shape[1] if audio.ndim > 1 else 1}")
        log_event(f"Duração: {len(audio) / sr:.2f} segundos")

        processed_audio = preprocess_audio(audio)
        monitoring_service.record_detected_events(db, session_id=active_session.id, audio_data=processed_audio)

        return MonitoringResponse(status="ok", detalhes="Áudio processado.")

    except Exception as e:
        log_event(f"Erro ao processar áudio da sessão {session_id}: {e}", to_file=True)
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")


@monitoring_router.get("/finalizar/{session_id}", response_model=SessionReportResponse)
def finalizar_monitoramento(
        session_id: int,
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user)
):
    """
    Finaliza uma sessão de monitoramento e retorna um relatório agregado
    dos eventos detectados.
    """
    session = db.query(monitoring_service.MonitoringSession).filter(
        session_id == monitoring_service.MonitoringSession.id,
        user_id == monitoring_service.MonitoringSession.user_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Sessão de monitoramento não encontrada.")

    if not session.end_time:
        session = monitoring_service.close_db_session(db, session)

    report = monitoring_service.generate_session_report(session)
    log_event(f"Sessão {session_id} finalizada. Relatório gerado.")
    return report


reports_router = APIRouter(prefix="/relatorios", tags=["Relatórios"])


def build_aggregated_report(sessions, start_date, end_date):
    """Função auxiliar para construir a resposta do relatório agregado."""
    report_items = monitoring_service.aggregate_sessions_data(sessions)

    total_tosse = sum(item['quantidade_tosse'] for item in report_items)
    total_espirro = sum(item['quantidade_espirro'] for item in report_items)
    total_outros = sum(item['outros_eventos'] for item in report_items)

    return AggregatedReport(
        periodo_inicio=start_date.strftime('%Y-%m-%d'),
        periodo_fim=end_date.strftime('%Y-%m-%d'),
        total_sessoes=len(sessions),
        total_tosse=total_tosse,
        total_espirro=total_espirro,
        total_outros_eventos=total_outros,
        sessoes=report_items
    )


@reports_router.get("/por_data", response_model=AggregatedReport)
def get_report_by_date(
        data: str = Query(...),  # Espera uma data no formato YYYY-MM-DD
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user)
):
    """Gera um relatório agregado de todas as sessões para uma data específica."""
    try:
        # Converte a string da data para um objeto datetime ciente do fuso horário
        selected_date = datetime.strptime(data, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de data inválido. Use YYYY-MM-DD.")

    day_start = selected_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    sessions = monitoring_service.get_sessions_by_date_range(db, user_id, day_start, day_end)
    return build_aggregated_report(sessions, day_start, day_end)


@reports_router.get("/semanal", response_model=AggregatedReport)
def get_weekly_report(
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user)
):
    """Gera um relatório agregado de todas as sessões dos últimos 7 dias."""
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    seven_days_ago = today_start - timedelta(days=7)

    sessions = monitoring_service.get_sessions_by_date_range(db, user_id, seven_days_ago,
                                                             today_start + timedelta(days=1))
    return build_aggregated_report(sessions, seven_days_ago, today_start)


router.include_router(auth_router)
router.include_router(monitoring_router)
router.include_router(reports_router)
