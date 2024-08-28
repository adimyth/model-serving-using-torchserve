# multi_model_tts_handler.py
import io
import json

import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel
from ts.torch_handler.base_handler import BaseHandler
from loguru import logger


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
        logger.debug(f"Received data: {data}")
        sentences = []
        languages = []
        sample_rates = []
        for d in data:
            # Check if the data is already a dict, if not, try to parse it
            if isinstance(d, dict):
                input_data = d
            else:
                input_data = json.loads(d.get("data") or d.get("body"))

            sentences.append(input_data.get("sentence"))
            languages.append(input_data.get("language", "en"))
            sample_rates.append(input_data.get("sample_rate", 22050))

        logger.debug(
            f"Preprocessed data: sentences={sentences}, languages={languages}, sample_rates={sample_rates}"
        )
        return sentences, languages, sample_rates

    def inference(self, data):
        sentences, languages, sample_rates = data
        outputs = []
        for sentence, language in zip(sentences, languages):
            model = self.models.get(language)
            tokenizer = self.tokenizers.get(language)
            if model is None or tokenizer is None:
                raise ValueError(
                    f"Model or tokenizer not found for language: {language}"
                )

            inputs = tokenizer(sentence, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output = model(**inputs)
            waveform = output.waveform[0].cpu().numpy()
            outputs.append(waveform)
        return outputs, sample_rates

    def postprocess(self, inference_output):
        waveforms, sample_rates = inference_output
        responses = []
        for waveform, sample_rate in zip(waveforms, sample_rates):
            buffer = io.BytesIO()
            sf.write(buffer, waveform, sample_rate, format="wav")
            buffer.seek(0)
            responses.append(buffer.getvalue())
        return responses

    def handle(self, data, context):
        try:
            logger.debug("Handling request")
            preprocessed_data = self.preprocess(data)
            inference_output = self.inference(preprocessed_data)
            return self.postprocess(inference_output)
        except Exception as e:
            logger.error(f"Error in handler: {str(e)}")
            raise
