# multi_model_tts_handler.py
import io
import json

import soundfile as sf
import torch
from loguru import logger
from transformers import AutoTokenizer, VitsModel
from ts.torch_handler.base_handler import BaseHandler


class MultiModelTTSHandler(BaseHandler):
    def __init__(self):
        super(MultiModelTTSHandler, self).__init__()
        self.initialized = False
        self.models = {}
        self.tokenizers = {}

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the HF_MODEL_DICT from a config file
        with open(f"{model_dir}/model_config.json", "r") as f:
            self.HF_MODEL_DICT = json.load(f)

        # Load models and tokenizers
        for lang, model_name in self.HF_MODEL_DICT.items():
            self.models[lang] = VitsModel.from_pretrained(model_name).to(self.device)
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)

        self.initialized = True

    def preprocess(self, data):
        # In our case, we will only have one request
        # [{'body': [{'sentence': '...', 'language': 'kn'}]}]
        data = data[0]["body"][0]
        sentence, language = data["sentence"], data["language"]
        logger.debug(f"Received TTS request:  ({language}) {sentence[:100]}")
        return sentence, language

    def inference(self, data):
        sentence, language = data
        model = self.models.get(language)
        tokenizer = self.tokenizers.get(language)
        if model is None or tokenizer is None:
            raise ValueError(f"Model or tokenizer not found for language: {language}")

        inputs = tokenizer(sentence, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = model(**inputs)
        waveform = output.waveform[0].cpu().numpy()
        return waveform

    def postprocess(self, waveform):
        buffer = io.BytesIO()
        sf.write(buffer, waveform, 16000, format="wav")
        buffer.seek(0)
        return buffer.getvalue()
