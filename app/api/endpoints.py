from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import soundfile as sf

from app.services.processing.audio_processing import preprocess_audio
from app.services.prediction_recognition.prediction import predict_sound
from app.services.visualization.visualization import generate_spectrogram
from app.services.log.logger import log_event
from app.db.session import get_db
from app.services.user import user_service

router = APIRouter()


@router.post("/register")
def register(name: str, email: str, password: str, db: Session = Depends(get_db)):
    existing = user_service.login_user(db, email, password)
    if existing:
        raise HTTPException(status_code=400, detail="Esse email já foi registrado")
    return user_service.register_user(db, name, email, password)


@router.post("/login")
def login(email: str, password: str, db: Session = Depends(get_db)):
    user = user_service.login_user(db, email, password)
    if user is None:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")
    return {"id": user.id, "name": user.name, "email": user.email}


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
