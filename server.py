import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
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
    logger.debug(f"Received TTS request: {request}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                TORCHSERVE_INFERENCE_URL,
                json=[
                    {  # Note the list wrapping here
                        "sentence": request.sentence,
                        "language": request.language,
                        "sample_rate": request.sample_rate,
                    }
                ],
                timeout=30.0,
            )
            logger.debug(f"TorchServe response status: {response.status_code}")
            logger.debug(f"TorchServe response content: {response.content[:100]}...")

            if response.status_code == 200:
                return StreamingResponse(response.iter_bytes(), media_type="audio/wav")
            else:
                logger.error(f"TorchServe error: {response.text}")
                raise HTTPException(
                    status_code=500, detail=f"TTS inference failed: {response.text}"
                )
        except httpx.RequestError as exc:
            logger.error(f"An error occurred while requesting {exc.request.url!r}.")
            raise HTTPException(status_code=500, detail=str(exc))
