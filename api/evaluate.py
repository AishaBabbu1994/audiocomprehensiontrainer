from fastapi import FastAPI, File, UploadFile, Form
import whisper
import tempfile
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

whisper_model = whisper.load_model("base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

    embeddings = embedder.encode(
        [text, transcription],
        convert_to_tensor=True
    )

    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    score = round(similarity * 100, 2)

    feedback = (
        "Excelente comprensión"
        if score >= 80 else
        "Buena comprensión, con omisiones"
        if score >= 60 else
        "Comprensión baja, requiere refuerzo"
    )

    return {
        "transcription": transcription,
        "score": score,
        "feedback": feedback
    }
