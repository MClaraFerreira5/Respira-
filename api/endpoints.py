from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import soundfile as sf
import numpy as np
import io

from services.processing.audio_processing import preprocess_audio
from services.prediction_recognition.prediction import predict_sound
from services.visualization.visualization import generate_spectrogram
from services.log.logger import log_event
from db.session import get_db
from services.user import user_service
from services.auth.auth import create_access_token
from api.dtos import (UserDto, LoginDto, SessionStartResponse, MonitoringResponse, SessionReportResponse)
from services import monitoring_service
from core.dependencies import get_current_user

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
def login(login: LoginDto, db: Session = Depends(get_db)):
    user = user_service.login_user(db, login.email, login.password)
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

# @monitoring_router.post('/analisar_audio')
# async def analisar_audio(file: UploadFile = File(...)):
#     try:
#         log_event("Recebendo arquivo de áudio")
#         contents = await file.read()

#         with open("audio_temp.wav", "wb") as f:
#             f.write(contents)
#         log_event("Arquivo salvo como audio_temp.wav")

#         audio, sr = sf.read("audio_temp.wav")
#         log_event(f"Áudio carregado (sample rate: {sr}, duração: {len(audio) / sr:.2f}s)")

#         processado = preprocess_audio(audio)
#         _, _, predicao = predict_sound(processado)
#         log_event("Predição realizada com sucesso")

#         imagem = generate_spectrogram(processado, predicao)
#         log_event(f"Espectrograma salvo como {imagem}")

#         return {"status": "ok", "espectrograma": imagem}

#     except Exception as e:
#         import traceback
#         log_event(f"Erro na análise de áudio: {str(e)}", to_file=True)
#         traceback.print_exc()
#         return {"status": "erro", "detalhes": str(e)}

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
    session_id: int,
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
        audio, sr = sf.read(io.BytesIO(contents))
        
        noise_profile = monitoring_service.get_noise_profile_from_audio(audio)
        monitoring_service.update_db_session_noise_profile(db, active_session, noise_profile)
        
        log_event(f"Ambiente da sessão {session_id} atualizado para '{noise_profile}'")
        
        return MonitoringResponse(status="ok", detalhes=f"Perfil de ruído atualizado para {noise_profile}.")
    except Exception as e:
        log_event(f"Erro ao analisar ambiente: {e}", to_file=True)
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")


@monitoring_router.post("/audio", response_model=MonitoringResponse)
async def monitorar_audio(
    session_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    """
    Recebe um trecho de áudio (loop), verifica se a sessão é válida
    e processa os eventos sonoros.
    """
    active_session = monitoring_service.get_active_db_session(db, session_id=session_id, user_id=user_id)
    if not active_session:
        raise HTTPException(status_code=404, detail="Sessão de monitoramento não encontrada, expirada ou inválida.")

    try:
        contents = await file.read()
        audio, sr = sf.read(io.BytesIO(contents))
        
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
        monitoring_service.MonitoringSession.id == session_id,
        monitoring_service.MonitoringSession.user_id == user_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Sessão de monitoramento não encontrada.")
    
    if not session.end_time:
        session = monitoring_service.close_db_session(db, session)
    
    report = monitoring_service.generate_session_report(session)
    log_event(f"Sessão {session_id} finalizada. Relatório gerado.")
    return report

router.include_router(auth_router)
router.include_router(monitoring_router)