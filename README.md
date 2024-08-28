## Serving Multiple HF Model using Torch Serve

1. Create a handler file. Refer [tts_handler.py](./tts_handler.py)
2. Create model archive file
```bash
# You will need HF API Key
source .env

# Create model store directory
mkdir model_store

# Create model archive file
torch-model-archiver --model-name multi_tts_model \
                     --version 1.0 \
                     --handler tts_handler.py \
                     --extra-files model_config.json \
                     --export-path ./model_store
```
3. Start Torch Serve
```bash
torchserve --start --ncs --model-store ./model_store --models multi_tts_model.mar --ts-config config.properties
```
4. Run the FastAPI server
```bash
uvicorn server:app --reload --port 8000 --host 0.0.0.0
```

To stop the Torch Serve
```bash
torchserve --stop
```