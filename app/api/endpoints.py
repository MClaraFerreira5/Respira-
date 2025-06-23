from fastapi import APIRouter, UploadFile, File

from app.services.audio_processing import preprocess_audio
from app.services.prediction import predict_sound
from app.services.visualization import generate_spectrogram
router = APIRouter()


@router.post('/analisar_audio')
async def analise_audio(file: UploadFile = File(...)):
    # Lê e armazena
    conteudo = await file.read()
    with open('audio_temp.wav', 'wb') as f:
        f.write(conteudo)

    # Processa
    from pydub import AudioSegment
    sound = AudioSegment.from_file("audio_temp.wav", format="wav")
    sound.export("audio_temp_fixed.wav", format="wav")

    import soundfile as sf
    audio, sr = sf.read('audio_temp.wav')
    processado = preprocess_audio(audio)
    _, _, predicao = predict_sound(audio)
    generate_spectrogram(processado, predicao)

    return {"status": "ok"}
