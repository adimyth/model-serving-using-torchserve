import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()


class TTSRequest(BaseModel):
    sentence: str
    language: str
    sample_rate: int = 22050


# Use environment variable for TorchServe URL, with a default value
TORCHSERVE_URL = os.getenv("TORCHSERVE_URL", "http://localhost:8080")
TORCHSERVE_INFERENCE_URL = f"{TORCHSERVE_URL}/predictions/multi_tts_model"


@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            TORCHSERVE_INFERENCE_URL,
            json={
                "sentence": request.sentence,
                "language": request.language,
                "sample_rate": request.sample_rate,
            },
        )
        if response.status_code == 200:
            return StreamingResponse(response.iter_bytes(), media_type="audio/wav")
        else:
            raise HTTPException(status_code=500, detail="TTS inference failed")
