from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import soundfile as sf

from services.processing.audio_processing import preprocess_audio
from services.prediction_recognition.prediction import predict_sound
from services.visualization.visualization import generate_spectrogram
from services.log.logger import log_event
from db.session import get_db
from services.user import user_service
from services.auth.auth import create_access_token
from api.dtos import UserDto, LoginDto
from services.log.logger import log_event
router = APIRouter()


@router.post("/register")
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


@router.post("/login")
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


@router.post('/analisar_audio')
async def analisar_audio(file: UploadFile = File(...)):
    try:
        log_event("Recebendo arquivo de áudio")
        contents = await file.read()

        with open("audio_temp.wav", "wb") as f:
            f.write(contents)
        log_event("Arquivo salvo como audio_temp.wav")

        audio, sr = sf.read("audio_temp.wav")
        log_event(f"Áudio carregado (sample rate: {sr}, duração: {len(audio) / sr:.2f}s)")

        processado = preprocess_audio(audio)
        _, _, predicao = predict_sound(processado)
        log_event("Predição realizada com sucesso")

        imagem = generate_spectrogram(processado, predicao)
        log_event(f"Espectrograma salvo como {imagem}")

        return {"status": "ok", "espectrograma": imagem}

    except Exception as e:
        import traceback
        log_event(f"Erro na análise de áudio: {str(e)}", to_file=True)
        traceback.print_exc()
        return {"status": "erro", "detalhes": str(e)}
