from fastapi import FastAPI
from pydantic import BaseModel
import re

app = FastAPI()

class Data(BaseModel):
    original: str
    transcription: str

def clean(text):
    return set(re.findall(r"\b\w+\b", text.lower()))

@app.post("/api/evaluate")
def evaluate(data: Data):
    original = clean(data.original)
    user = clean(data.transcription)

    if not original:
        score = 0
    else:
        score = round(len(original & user) / len(original) * 100)

    feedback = (
        "Excelente comprensión"
        if score >= 80 else
        "Comprensión aceptable"
        if score >= 60 else
        "Comprensión baja"
    )

    return {
        "score": score,
        "feedback": feedback
    }
