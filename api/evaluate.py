from fastapi import FastAPI, File, UploadFile, Form
import whisper
import tempfile
import re

app = FastAPI()

whisper_model = whisper.load_model("tiny")

def clean(text):
    return re.findall(r"\b\w+\b", text.lower())

@app.post("/api/evaluate")
async def evaluate(
    text: str = Form(...),
    audio: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await audio.read())
        audio_path = tmp.name

    result = whisper_model.transcribe(audio_path)
    transcription = result["text"]

    original_words = set(clean(text))
    user_words = set(clean(transcription))

    if not original_words:
        score = 0
    else:
        overlap = original_words.intersection(user_words)
        score = round((len(overlap) / len(original_words)) * 100)

    feedback = (
        "Excelente comprensión del texto"
        if score >= 80 else
        "Comprensión aceptable, puede mejorar"
        if score >= 60 else
        "Comprensión baja, necesita refuerzo"
    )

    return {
        "transcription": transcription,
        "score": score,
        "feedback": feedback
    }
